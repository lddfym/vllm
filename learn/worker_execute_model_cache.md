# Worker.execute_model GPU Cache 异步传输机制

基于 `vllm/worker/worker.py`、`vllm/worker/cache_engine.py`、`csrc/cache_kernels.cu` 源码分析。

---

## 0. 前置知识

### 0.1 CUDA 异步执行与跨流同步

**CUDA 操作是异步的**。CPU 调用 CUDA API 时，只是把指令塞入 stream 的命令队列，立即返回，不等 GPU 实际执行完毕。

```python
result = large_tensor @ large_tensor.T  # CPU 立即返回
# result 是 tensor 对象（有形状、类型、GPU 内存地址），
# 但 GPU 可能还未完成计算，数值尚未写入内存。
# 直到某个同步点（如 .cpu()、event.synchronize()）之前，数值不可靠。
```

**跨流同步三个 API**：

| API | 参数默认值 | CPU 是否阻塞 | 作用 |
|---|---|---|---|
| `event.record(stream=None)` | 当前 stream | 否 | 向指定 stream 的命令队列插入一个**同步标记**。GPU 执行到该标记时，将 event 状态置为"已完成" |
| `event.wait(stream=None)` | 当前 stream | 否 | 向指定 stream 的命令队列插入"等待检查"指令。GPU 执行到该指令时，若 event 未完成则暂停该 stream，完成后继续 |
| `event.synchronize()` | — | **是** | CPU 线程阻塞，直到 event 变为"已完成" |

`record` 和 `wait` 均只是向命令队列**插入指令**，CPU 立即返回；实际等待发生在 **GPU 侧**。

```
cache_stream:   ... [swap_blocks layer i] ──[record → event[i].state=done]──▶
                                                  ↑ GPU 执行到同步标记时触发
default stream: ... [wait → 检查 event[i].state] ──等到 done 后继续──▶ [attn kernel layer i]
                          ↑ CPU 已返回，GPU 侧等待
```

两个关键性质：
- **一次 record 可被多次 wait**：event 完成后保持"已完成"状态，直到下一次 `record()` 重置
- **同一 stream 内无需 event**：stream 内命令严格串行；event 仅用于**跨 stream 同步**

### 0.2 PyTorch CUDA Caching Allocator：流序延迟释放

GPU tensor 析构后，底层 GPU 内存不会立即释放，而是走 PyTorch CUDA Caching Allocator 的流序延迟释放（`c10/cuda/CUDACachingAllocator.cpp`）。

tensor 引用计数降为 0 时，析构器调用 allocator `free()`，但**不直接调用 `cudaFree`**：

```cpp
void free(void* ptr) {
    Block* block = find_block(ptr);
    cudaEventRecord(block->event, block->stream);  // 在关联 stream 上插入检查点
    pending_free_list.push(block);                  // 放入 pending-free 池
    // 等 event 触发后，下次 malloc lazy 检查时才真正归还
}
```

**时机为何是对的**：C++ 顺序语义保证 kernel 提交先于析构器触发，检查点 event 天然落在 kernel 之后：

```
default stream: [...][kernel using tensor memory]...[event record by free()]...
                         ↑ 使用内存                    ↑ event 触发 ⟺ kernel 已完成
                                                        内存才可被重新分配
```

**lazy 回收**：event 触发后不立刻归还——下次 `malloc` 空闲块不足时扫描 pending-free 列表，`cudaEventQuery(event) == cudaSuccess` 的 block 才归还空闲池。

| 层 | 负责什么 |
|---|---|
| **PyTorch**（`CUDACachingAllocator`） | block→stream 映射、pending-free 列表、lazy 检查逻辑 |
| **CUDA** | `cudaEventRecord` / `cudaEventQuery` / `cudaMalloc` / `cudaFree` 原语 |

> CUDA 11.2+ 引入了 `cudaMallocAsync` / `cudaFreeAsync`，可在驱动层实现流序分配；PyTorch 默认 allocator 在用户态自行实现了这套逻辑。

---

## 1. 双流设计

`CacheEngine.__init__`（`cache_engine.py:44`）创建两个独立资源：

```
cache_stream = torch.cuda.Stream()               ← 专用于 KV cache 迁移
events       = [torch.cuda.Event() × num_layers] ← 每层一个跨流同步事件
```

目的：让 KV cache 数据迁移与模型计算**重叠**：

```
cache_stream:   [swap layer 0][record 0] [swap layer 1][record 1] [swap layer 2][record 2] ...
                                  ↓ event[0] done          ↓ event[1] done         ↓ event[2] done
default stream:    [wait 0][attn layer 0] [wait 1][attn layer 1] [wait 2][attn layer 2] ...
```

layer i 的 attention 等到 event[i] 触发后开始，此时 cache_stream 已在传输 layer i+1，实现**逐层流水线**。

---

## 2. events 的 record / wait 使用模式

### 2.1 record：swap 每层完成后打点

`CacheEngine._swap()`（`cache_engine.py:108`）：

```python
with torch.cuda.stream(self.cache_stream):
    for i in range(self.num_layers):
        cache_ops.swap_blocks(src_key_cache, dst_key_cache, src_to_dst)
        cache_ops.swap_blocks(src_value_cache, dst_value_cache, src_to_dst)
        self.events[i].record(stream=self.cache_stream)   # 第 i 层 swap 完成后打点
```

`events[i]` 触发时机：cache_stream 完成第 i 层的 swap_blocks kernel。

### 2.2 wait：model 每层 attention 前检查

**有 input 时**（`worker.py:279`）：`cache_events` 传入 `self.model()`，模型在每层 attention 读 KV cache 前调用 `events[i].wait()`（无参数，插入 default stream），确保该层 swap 完成后才读。

**无 input 时**（`worker.py:267`）：模型不被调用，直接循环调用 `event.wait()` 向 default stream 插入屏障（详见第 3 节）。

### 2.3 互斥保证：同一 step 只有一次 record

调度器 swap-in 阶段的条件（`scheduler.py`）：

```python
while self.swapped and not blocks_to_swap_out:   # 有 swap_out 则不做 swap_in
    ...swap_in...
```

同一 step 内 swap_in 与 swap_out 互斥，`self.events`（共 `num_layers` 个）只会被一次 `_swap()` 调用 record，不存在争抢问题。

---

## 3. 无 input 时的同步兜底

```python
if not seq_group_metadata_list:
    if cache_events is not None:
        for event in cache_events:
            event.wait()    # 不传参，插入 default stream
    return {}
```

**触发场景**：调度器检查 running 队列时，因 GPU 显存不足，所有 running 序列全部被 SWAP 抢占（`running` 清空 → `seq_group_metadata_list` 为空），但 `blocks_to_swap_out` 非空。这要求所有 running 序列均为多序列 beam search（`best_of > 1`）：单序列（`best_of=1`）被抢占走 RECOMPUTE 路径（直接释放 GPU 块，不产生 `blocks_to_swap_out`），只有 beam search 走 SWAP 路径（GPU→CPU，产生 `blocks_to_swap_out`）。

**为什么不能省略**：swap_out 在 cache_stream 上异步读 GPU 块（如 block 42），调度器已将 block 42 标记为逻辑空闲。下一步 block 42 可能被新序列分配，model forward 在 default stream 上向 block 42 写入新 KV 数据。若未同步 → 数据竞争：

```
Step N（无 input）：cache_stream 异步读 block 42；调度器标记 block 42 空闲
Step N+1（有 input）：block 42 分配给新序列；default stream 向 block 42 写入
                       若 Step N 未插屏障 → 读/写并发 → 静默错误
```

`event.wait()` 在 default stream 中插入屏障后立即返回。屏障持续到下一步 model forward 开始前：

```
default stream: ─[wait 0][wait 1]...── Step N 返回 ──── Step N+1: model kernels ──▶
                     ↑ swap_out 完成前，GPU 不执行后续 kernel
```

---

## 4. copy() 与 swap() 的设计差异

`execute_model` 除了处理 swap_in/swap_out，还处理 **CoW（Copy-on-Write）** 操作。beam search 中多条序列共享同一 prompt 末块（`ref_count > 1`），首次 decode 时 `_append_slot` 检测到共享，触发 CoW：分配新私有块，将旧共享块数据拷贝到新块。这个拷贝通过 `CacheEngine.copy()` / `copy_blocks` 完成。

`copy_blocks` 与 `swap_blocks` 的实现路径截然不同，以下逐维度分析。

### 4.1 底层实现对比（`csrc/cache_kernels.cu`）

| 维度 | `copy_blocks`（CoW） | `swap_blocks`（swap in/out） |
|---|---|---|
| 处理粒度 | **一次 kernel launch 处理所有 layer**<br/>2D grid: `dim3(num_layers, num_pairs)` | 每次调用处理单个 layer 的一个 tensor<br/>（Python 层 `for i in range(num_layers)` 循环） |
| 实现方式 | 自定义 CUDA kernel | `cudaMemcpyAsync` 循环 |
| 数据路径 | GPU 内部拷贝（gpu_cache → gpu_cache） | GPU ↔ CPU（PCIe 传输） |
| 使用 stream | **default stream**（`at::cuda::getCurrentCUDAStream()`） | `cache_stream` |
| 同步方式 | kernel launch 前 `.to(cache_device)` 以 `cudaMemcpy`（同步，CPU 阻塞）上传 CPU 侧临时指针数组（见下方注）；kernel 本身异步提交到 default stream，与 model forward 同流串行 | 完全异步，靠 `event.record/wait` 跨流同步 |

> **`.to(cache_device)` 为何是同步点**：`key_cache_ptrs` 是 CPU 侧临时数组（非 pinned 内存），`from_blob` 包裹此内存（不复制）。`.to(cache_device)` 触发 CPU→GPU 传输：
> - 默认 `non_blocking=False` → `cudaMemcpy`（同步）：CPU 阻塞直到传输完成。
> - 即使设 `non_blocking=True`，对非 pinned 内存，CUDA 规范规定 `cudaMemcpyAsync` 行为等同 `cudaMemcpy`，CPU 仍然阻塞。（只有 pinned 内存才能实现真正的 async H2D 传输；而即使那样，由于 async copy 和 kernel 均在同一 default stream 上，stream 内串行仍保证 copy 先于 kernel 完成。）
>
> 这正是源码注释 `// This synchronizes the CPU and GPU` 的含义。传输完成后 `key_cache_ptrs_tensor` 成为有效 GPU tensor；`copy_blocks_kernel<<<>>>` 随后在 default stream 上异步使用此 GPU buffer；函数返回后 GPU 内存通过前置知识 0.2 的流序延迟释放安全回收。

### 4.2 为什么两者设计不同

- **swap**：PCIe 传输慢（~30 GB/s），逐层执行 + per-layer event 让 model 计算 layer i 时，cache_stream 并发传输 layer i+1，实现流水线。必须跨流同步，event 不可省。
- **copy**：GPU 内部带宽高（~2 TB/s），一次 kernel 所有层并发执行。copy kernel 与 model forward 同在 default stream，天然串行有序，无需跨流 event。

### 4.3 issued_cache_op = True 对 copy 是 bug

```python
if blocks_to_copy:
    self.cache_engine.copy(blocks_to_copy)
    issued_cache_op = True      # ← bug：copy 不 record events，设 True 是错的
```

`copy_blocks` 不走 `cache_stream`，不调用 `event.record()`。设 `issued_cache_op = True` 导致无关的 events 被传给 model，model 调用 `event.wait()`。

**为什么没崩**：`torch.cuda.Event` 是懒初始化（`c10/cuda/CUDAEvent.h`），event handle 仅在第一次 `record()` 时创建。`event.wait()` 的 C++ 实现有 guard：

```cpp
// PyTorch 内部对应 Python 的 event.wait()，"block" 指阻塞 GPU stream，不阻塞 CPU
void block(const CUDAStream& stream) {
    if (is_created_) {
        cudaStreamWaitEvent(stream, event_, 0);
    }
    // is_created_=false 时直接返回，不调用任何 CUDA API
}
```

copy-only 步骤（有 CoW 但无 swap）在服务生命周期中可多次出现：每个 beam search 请求首次 decode 时，若内存充足不触发抢占，则只有 CoW 而无 swap。两种情况下 `event.wait()` 均立即返回，不阻塞：
- **events 从未 record 过**（服务启动后还未发生任何 swap）：`is_created_ = false`，直接返回
- **events 已被之前 swap record 并触发**：已完成的 event 不会阻塞，一次 record 可被多次 wait

**结论**：功能上无害的代码错误，copy 数据通过 default stream 的串行保证已就位，与 events 无关。

### 4.4 copy_blocks 的 GPU 内存安全

`copy_blocks` 函数返回后 `key_cache_ptrs_tensor` 析构，但 `copy_blocks_kernel` 可能尚未执行完毕。GPU 内存不会被提前回收——依赖前置知识 0.2 的流序延迟释放：C++ 顺序语义保证 kernel 提交先于析构，allocator `free()` 在 default stream 上记录检查点 event，event 触发后内存才可重新分配。

---

## 5. 完整时序图

两个典型场景：

**场景 A：有 input + swap_out + CoW**（部分序列被抢占，部分序列正常 decode 触发 CoW）

```
execute_model(seq_list=[...], swap_out={42:5}, blocks_to_copy={7:[8]})
│
├─ cache_engine.swap_out({42:5})
│     └─ cache_stream：
│           [swap key/val layer 0] [record event[0]]
│           [swap key/val layer 1] [record event[1]] ...
│
├─ copy_blocks({7:[8]})
│     ├─ .to(cache_device)（含 key/value 指针和 block_mapping）← CPU 等待上传（同步点）
│     └─ copy_blocks_kernel<<<>>>          ← kernel 异步提交到 default stream
│
├─ cache_events = self.cache_events        ← issued_cache_op=True（swap_out 已设置）
│
└─ self.model(..., cache_events=cache_events)
      default stream：
      [copy kernel]                                   ← 上文已提交
      [wait event[0]] → [attn layer 0]                ← swap layer 0 完成后执行
      [wait event[1]] → [attn layer 1] ...            ← swap layer 1 完成后执行
      (cache_stream 同时在传输 layer i+1，与 attn layer i 重叠)
```

**场景 B：无 input + swap_out**（所有 beam search 序列全部被 SWAP 抢占）

```
Step N：execute_model(seq_list=[], swap_out={42:5, 13:3}, blocks_to_copy={})
│
├─ cache_engine.swap_out({42:5, 13:3})
│     └─ cache_stream：[swap layer 0][record event[0]][swap layer 1][record event[1]] ...
│
└─ seq_list 为空 → for event in cache_events: event.wait()   ← 屏障插入 default stream
                  → return {}

      default stream 队列（此时仍保留屏障）：
      ─[wait event[0]][wait event[1]]...─────── Step N+1 model kernels ──▶
                                          ↑ swap_out 完成后屏障自动解除
```

---

## 关键文件索引

| 位置 | 文件 | 行号 |
|---|---|---|
| `Worker.execute_model` | `vllm/worker/worker.py` | L243 |
| `CacheEngine.__init__`（cache_stream / events） | `vllm/worker/cache_engine.py` | L43-47 |
| `CacheEngine._swap`（per-layer record） | `vllm/worker/cache_engine.py` | L102 |
| `CacheEngine.copy`（default stream，懒同步） | `vllm/worker/cache_engine.py` | L127 |
| `copy_blocks` CUDA kernel（2D grid 所有层一次 launch） | `csrc/cache_kernels.cu` | L52 |
| `swap_blocks` 实现（`cudaMemcpyAsync` 逐 block） | `csrc/cache_kernels.cu` | — |
| `torch.cuda.Event` 懒初始化 guard | `c10/cuda/CUDAEvent.h` | — |
| `CUDACachingAllocator`（流序延迟释放） | `c10/cuda/CUDACachingAllocator.cpp` | — |
