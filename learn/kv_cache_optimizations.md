# vLLM KV Cache 工程优化

## 一、多 Worker 场景的 Block 对齐

### 为什么有多个 Worker

vLLM 支持两种并行策略（通过 `ParallelConfig` 配置）：
- **张量并行（Tensor Parallelism）**：将模型 Attention 的 head 拆分到多个 GPU，每个 GPU 负责 `total_heads / tensor_parallel_size` 个 head，共同计算同一批 token。
- **流水线并行（Pipeline Parallelism）**：将模型的层拆分到不同 GPU，每个 GPU 负责 `total_layers / pipeline_parallel_size` 层（当前版本尚未实现，代码中有 `NotImplementedError` 保护）。

Worker 总数 = `pipeline_parallel_size × tensor_parallel_size`，每个 worker 独占一个 GPU，`world_size > 1` 时以 Ray Actor（独立进程）运行。

### 每个 Worker 的 Cache

每个 worker 独立持有自己的 `CacheEngine`，在本地 GPU 上分配 `gpu_cache`，在本地 CPU 上分配 `cpu_cache`。张量并行下，TP 按 **head 数**拆分（而非 head_dim），每个 worker 持有 `total_heads / tensor_parallel_size` 个完整 head（`head_size` 不变）。

### 问题与解决方案

各 GPU 可用显存可能不同，若各 worker 独立计算 `num_gpu_blocks`，结果会不一致。LLMEngine 使用**集中式调度器**，维护统一的 block 地址空间，调度器分配的 block ID（即 `gpu_cache[layer]` 数组下标）在所有 worker 间共用。若某 worker 的 block 数量少于调度器分配的最大 ID，将发生越界访问。

因此 `vllm/engine/llm_engine.py` 收集所有 worker 的计算结果后取最小值，再统一下发：

```python
# llm_engine.py _init_cache
num_blocks = self._run_workers("profile_num_available_blocks", get_all_outputs=True, ...)
num_gpu_blocks = min(b[0] for b in num_blocks)  # 取最小值，确保所有 worker 地址空间一致
num_cpu_blocks = min(b[1] for b in num_blocks)
self._run_workers("init_cache_engine", cache_config=self.cache_config)
```

---

## 二、专用 CUDA Stream 与 Event 同步

### 背景

每一步推理前，调度器可能需要将若干 block 在 CPU 和 GPU 之间传输（swap in/out）。若传输在默认 compute stream 上串行执行，GPU 在 PCIe 传输期间完全空闲。

### CUDA Stream 与 Event

CUDA Stream 是 GPU 上的命令队列，同一 stream 内的操作顺序执行，不同 stream 间可并行。PyTorch 有一个默认的 **compute stream**（`current_stream()`），所有模型计算在其上执行。

`Event` 是跨 stream 的同步原语：
- `event.record(stream=A)`：在 stream A 中插入标记，表示"此前的操作已完成"
- `event.wait()`（在 compute stream 上调用）：向 compute stream 插入 GPU 侧屏障，GPU 执行到此处时等待 Event 触发，**不阻塞 CPU**

### 实现

`CacheEngine` 初始化时创建独立的 `cache_stream` 和每层一个 `Event`：

```python
# cache_engine.py
self.cache_stream = torch.cuda.Stream()
assert self.cache_stream != torch.cuda.current_stream()  # 若相同则 event.wait() 等待自身，造成死锁
self.events = [torch.cuda.Event() for _ in range(self.num_layers)]
```

swap 操作投递到 `cache_stream`，每完成一层记录该层的 Event：

```python
def _swap(self, src, dst, src_to_dst):
    with torch.cuda.stream(self.cache_stream):
        for i in range(self.num_layers):
            cache_ops.swap_blocks(...)
            self.events[i].record(stream=self.cache_stream)
```

compute stream 在第 i 层 Attention 前等待**第 i 层**的 Event：

```python
# attention.py
if cache_event is not None:
    cache_event.wait()  # GPU 侧屏障，CPU 不阻塞
```

### 并行原理

CUDA Stream 的操作均为**异步**：CPU 将任务投入 stream 后立即返回，GPU 在后台执行。`execute_model`（`worker.py`）中 Python 顺序调用：

```python
self.cache_engine.swap_in(blocks_to_swap_in)   # 投入 cache_stream，CPU 立即返回
self.model(..., cache_events=cache_events)      # 投入 compute stream，CPU 立即返回
```

两次调用返回后，GPU 上两个 stream 同时运行：`cache_stream` 传输后续层数据，`compute stream` 计算当前层。`event.wait()` 保证第 i 层数据就绪后 compute stream 才继续。

---

## 三、CPU Cache 使用 Pin Memory

### Pageable Memory 与 Pinned Memory

操作系统以「页」为单位管理内存：

- **Pageable Memory（可分页内存）**：默认分配方式（`malloc`）。操作系统可将这些页换出到磁盘以释放物理内存，物理地址不固定。
- **Pinned Memory（页锁定内存）**：通过 `cudaHostAlloc` 分配，操作系统保证这段内存永久驻留物理内存，不会被换出。代价是占用不可换出的物理内存，过度使用会挤压系统整体可用内存。

适用场景：需要频繁在 CPU 与 GPU 之间高性能传输的缓冲区（如 swap 缓冲区）使用 Pinned Memory；普通 CPU 计算数据使用 Pageable Memory 即可。

### 为何 Pageable Memory 传输效率低

GPU DMA 引擎无法直接访问物理地址不固定的 Pageable Memory。发起传输时，CUDA 驱动需先将数据 memcpy 到一块临时的 pinned staging buffer，再由 DMA 引擎将 staging buffer 传输到 GPU，存在一次多余的 CPU 侧拷贝。

### 解决方案

CPU cache 分配时使用 `pin_memory=True`，GPU DMA 引擎可直接读写，无需 staging buffer：

```python
# cache_engine.py
# CPU cache：用于 swap，需要频繁 CPU↔GPU 传输
key_blocks = torch.empty(size=(...), dtype=self.dtype, pin_memory=True)
# GPU cache：本身在显存中，无需 pin_memory
key_blocks = torch.empty(size=(...), dtype=self.dtype, device="cuda")
```

### 两个好处

1. **更高带宽**：消除 staging copy。CUDA Best Practices Guide："Page-locked or pinned memory transfers attain the highest bandwidth between the host and the device."
2. **真正异步的 DMA**：同一文档："the asynchronous transfer version requires pinned host memory"，"the overlap once again requires pinned host memory, and, in addition, the data transfer and kernel must use different, non-default streams."  
   Pageable Memory 下异步传输无法生效——这是第二节 stream 并行的前提。  
   （引用：[CUDA C++ Best Practices Guide — Pinned Memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#pinned-memory)）

---

## 关键文件

| 文件 | 相关内容 |
|---|---|
| `vllm/engine/llm_engine.py` | `_init_cache`、`_run_workers` |
| `vllm/worker/worker.py` | `init_cache_engine`、`execute_model`、`profile_num_available_blocks` |
| `vllm/worker/cache_engine.py` | `__init__`、`_swap`、`allocate_cpu_cache`、`allocate_gpu_cache` |
| `vllm/model_executor/layers/attention.py` | `cache_event.wait()` |
| `vllm/config.py` | `ParallelConfig`（tensor/pipeline_parallel_size） |
