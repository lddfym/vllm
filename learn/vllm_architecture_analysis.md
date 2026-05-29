# vLLM 技术架构深度分析报告

基于当前 commit 的源码分析，以下是 vLLM 的完整技术架构报告：

---

## 一、项目整体架构

```
vllm/
├── core/              # 核心调度和内存管理
│   ├── scheduler.py    # 请求调度器
│   ├── block_manager.py # KV Cache块管理器
│   └── policy.py      # 调度策略
├── engine/            # 推理引擎
│   ├── llm_engine.py  # 主引擎
│   └── async_llm_engine.py # 异步引擎
├── worker/            # GPU Worker
│   ├── worker.py       # 工作进程
│   └── cache_engine.py # KV缓存引擎
├── model_executor/    # 模型执行器
│   ├── layers/
│   │   └── attention.py    # PagedAttention层
│   └── models/       # 支持的模型
├── entrypoints/       # API入口
└── csrc/             # CUDA内核
    ├── attention/     # Attention CUDA kernels
    └── cache_kernels.cu  # Cache CUDA kernels
```

---

## 二、核心组件设计

### 2.1 LLMEngine ([llm_engine.py:20](../vllm/engine/llm_engine.py#L20))

主引擎负责协调整个推理流程：

```python
class LLMEngine:
    - workers: List[Worker]           # GPU Worker进程池
    - scheduler: Scheduler             # 请求调度器
    - tokenizer                       # 分词器
    - seq_counter                    # 序列ID计数器
```

**关键方法**：
- `step()` - 执行一次解码迭代，核心循环
- `add_request()` - 添加新请求
- `_run_workers()` - 并行执行Worker

### 2.2 Scheduler ([scheduler.py:51](../vllm/core/scheduler.py#L51))

三状态队列调度器：

```python
class Scheduler:
    - waiting: List[SequenceGroup]   # 等待队列
    - running: List[SequenceGroup]   # 运行队列
    - swapped: List[SequenceGroup]   # 换出队列
    - block_manager: BlockSpaceManager
```

**调度流程** ([scheduler.py:105-257](../vllm/core/scheduler.py#L105-L257))：
1. 优先处理 RUNNING 状态的序列
2. 为序列分配新token槽位
3. 内存不足时进行抢占
4. 尝试换入 SWAPPED 状态的序列
5. 尝试启动 WAITING 状态的新请求

### 2.3 BlockManager ([block_manager.py:9](../vllm/core/block_manager.py#L9))

**核心数据结构**：

| 类型 | 说明 |
|------|------|
| `LogicalTokenBlock` | 逻辑块，存储token ID序列 |
| `PhysicalTokenBlock` | 物理块，存储KV缓存，带引用计数 |
| `BlockTable` | 逻辑块到物理块的映射表 |

**关键特性**：
- 引用计数管理内存共享
- Copy-on-Write 机制处理共享块的修改
- GPU/CPU 双层缓存支持换入换出

---

## 三、PageAttention 核心技术

### 3.1 设计理念

PageAttention 借鉴操作系统的虚拟内存分页机制，将 KV cache 划分为固定大小的块（如16个token），实现：

1. **非连续存储**：逻辑token序列可以映射到非连续的物理块
2. **按需分配**：仅在需要时分配物理块
3. **共享内存**：多个序列可以共享相同的物理块（beam search时）

### 3.2 核心实现 ([attention.py:16](../vllm/model_executor/layers/attention.py#L16))

```python
class PagedAttention(nn.Module):
    def forward(self, query, key, value, key_cache, value_cache,
               input_metadata, cache_event):
        # 1. 处理prompt阶段（使用xformers优化）
        if num_prompt_tokens > 0:
            self.multi_query_kv_attention(...)

        # 2. 等待cache操作完成
        if cache_event is not None:
            cache_event.wait()

        # 3. 将新的KV写入cache
        if key_cache is not None:
            cache_ops.reshape_and_cache(key, value, ...)

        # 4. 处理生成阶段（从cache读取KV）
        if num_generation_tokens > 0:
            self.single_query_cached_kv_attention(...)
```

### 3.3 CUDA Kernel 优化 ([attention_kernels.cu:74](../csrc/attention/attention_kernels.cu#L74))

**single_query_cached_kv_attention_kernel** 核心优化点：

```cpp
// Grid: (num_heads, num_seqs)
template<typename scalar_t, int HEAD_SIZE, int BLOCK_SIZE, int NUM_THREADS>
__global__ void single_query_cached_kv_attention_kernel(...)
```

**优化策略**：
1. **Thread Group**：每个warp内多个线程协作处理一个token的key，实现向量化加载
2. **Warp-level reduction**：使用 `__shfl_xor_sync` 进行warp内归约
3. **Shared Memory**：存储softmax中间结果
4. **Flash Attention风格**：在线性时间内计算attention，避免完整attention矩阵存储

**数据布局优化**：
```
Key Cache: [num_blocks, num_heads, head_size/x, block_size, x]
Value Cache: [num_blocks, num_heads, head_size, block_size]
```
这种布局优化了GPU内存访问模式，通过x维实现向量化加载。

### 3.4 Cache 操作内核

**reshape_and_cache_kernel** ([cache_kernels.cu:143](../csrc/cache_kernels.cu#L143))：
- 将QKV向量reshape并写入分页cache
- 使用 `slot_mapping` 将token索引映射到物理cache位置

**copy_blocks_kernel** ([cache_kernels.cu:53](../csrc/cache_kernels.cu#L53))：
- 跨层并行复制blocks（Grid: num_layers × num_pairs）
- 处理Copy-on-Write场景

**swap_blocks**：
- GPU↔CPU内存异步传输
- 支持换入换出策略

---

## 四、数据流转完整流程

```
用户请求 (prompt)
    ↓
[LLMEngine.add_request]
    ↓ tokenizer → prompt_token_ids
    ↓ 创建Sequence和SequenceGroup
    ↓
[Scheduler.add_seq_group] → waiting队列
    ↓
[Engine.step] ← 主循环
    ↓
[Scheduler.schedule]
    ├─ [BlockManager] 内存分配/释放
    ├─ [BlockManager] 换入/换出
    └─ 返回 seq_group_metadata_list
    ↓
[Worker.execute_model]
    ├─ 准备输入张量 (token_ids, positions, slot_mapping)
    ├─ 执行cache操作 (swap_in/swap_out/copy)
    ├─ [模型forward]
    │   └─ [PagedAttention.forward]
    │       ├─ prompt: multi_query_kv_attention (xformers)
    │       ├─ cache_ops.reshape_and_cache (写入KV)
    │       └─ generation: single_query_cached_kv_attention
    └─ 返回 seq_outputs
    ↓
[Scheduler.update] 更新序列状态
    ↓
[Engine._decode_sequences] 增量解码
    ↓
返回 RequestOutput
```

**关键数据结构流转**：

```
Sequence (用户视角)
    ↓ logical_token_blocks [LogicalTokenBlock, ...]
    ↓ block_tables (seq_id → [PhysicalBlock, ...])
    ↓ slot_mapping (token_idx → physical_slot)
    ↓ KV Cache [num_blocks, num_heads, ...]
```

---

## 五、核心性能优化点

### 5.1 内存管理优化

| 优化项 | 实现位置 | 效果 |
|--------|-----------|------|
| PagedAttention | [attention.py](../vllm/model_executor/layers/attention.py) | 消除内存碎片 |
| 引用计数 | [block.py:62](../vllm/block.py#L62) | beam search共享内存 |
| CoW机制 | [block_manager.py:132](../vllm/core/block_manager.py#L132) | 减少内存复制 |
| CPU/GPU交换 | [cache_engine.py:121](../vllm/worker/cache_engine.py#L121) | 支持更多并发请求 |

### 5.2 计算优化

| 优化项 | 实现位置 | 说明 |
|--------|-----------|------|
| 动态批处理 | [scheduler.py:173](../vllm/core/scheduler.py#L173) | prompt和generation混合 |
| 向量化加载 | [attention_kernels.cu:98](../csrc/attention/attention_kernels.cu#L98) | 16字节对齐 |
| Warp归约 | [attention_kernels.cu:42](../csrc/attention/attention_kernels.cu#L42) | 减少__syncthreads |
| Tensor Cores | [worker.py:214](../vllm/worker/worker.py#L214) | 8倍对齐填充 |

### 5.3 调度优化

**抢占策略** ([scheduler.py:343](../vllm/core/scheduler.py#L343))：
- **Recompute**：单序列场景，丢弃后重算
- **Swap**：多序列场景（beam search），换出到CPU

**水印机制** ([block_manager.py:64](../vllm/core/block_manager.py#L64))：
```python
watermark_blocks = int(watermark * num_gpu_blocks)
```
保留部分空闲blocks，避免频繁触发抢占。

### 5.4 CUDA Kernel 特殊优化

1. **循环展开**：`#pragma unroll` 减少分支预测
2. **LDG指令**：`__ldg` 通过只读缓存加载数据
3. **共享内存复用**：logits和output共享shared memory
4. **模板特化**：针对不同head_size和block_size编译专用kernel

---

## 六、技术亮点总结

1. **PageAttention**：操作系统分页思想在KV cache上的创新应用
2. **迭代级调度**：每个step重新调度，最大化GPU利用率
3. **内核级优化**：精心设计的CUDA kernel实现接近硬件极限的性能
4. **弹性内存管理**：GPU/CPU交换机制支持突发流量
5. **模块化设计**：清晰的分层架构，易于扩展新模型和调度策略

---

*基于 commit 67d96c29f 的源码分析*
