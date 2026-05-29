# vLLM KV Cache Block 机制

## 一、Block 的定义

在 Transformer 推理中，每一层均包含一个 Attention 子层。每个 token 经过每一层 Attention 计算后会产生 key 和 value 向量，为避免自回归生成时重复计算，这些向量需要被缓存（KV Cache），因此每个 token 在每一层都有独立的 key/value 缓存。

vLLM 借鉴操作系统分页内存的思想，以 **block** 为单位管理 KV Cache 的显存分配。一个 block 存放 `block_size` 个连续 token 在所有层（每层各一份 key + value）的 KV Cache，是显存分配的最小单元。

每个 block 在 GPU 上的 tensor 形状（每层）：

```
Key Cache block：   [num_heads, head_size // x, block_size, x]
Value Cache block： [num_heads, head_size, block_size]

x = 16 // element_size_bytes（fp16 → x=8，fp32 → x=4）
```

> Key Cache 中 `x` 维是为了 CUDA 16-byte 向量化加载而做的维度重排，不影响元素总数。

---

## 二、单个 Block 的字节数

单个 block 的字节数由 `vllm/worker/cache_engine.py` 中的 `get_cache_block_size` 计算：

```python
# 单层中，一个 block 的 key（或 value）元素数
#（元素指单个浮点数，如 fp16/fp32 的一个数值）
key_cache_block   = block_size * num_heads * head_size
value_cache_block = key_cache_block  # key/value 形状不同，但元素总数相同

total       = num_layers * (key_cache_block + value_cache_block)
block_bytes = dtype_size * total
```

各变量含义：

| 变量 | 含义 |
|---|---|
| `block_size` | 每个 block 存放的 token 数（如 16） |
| `num_heads` | 每 GPU 的 attention head 数 = `total_heads // tensor_parallel_size` |
| `head_size` | 每个 head 的向量维度 = `hidden_size // num_attention_heads` |
| `num_layers` | 每 GPU 的层数 = `total_layers // pipeline_parallel_size` |
| `dtype_size` | 每个元素的字节数（fp16=2，fp32=4） |

以 LLaMA-7B（fp16，单卡，`block_size=16`）为例：

- `head_size = 4096 // 32 = 128`
- 单层 key 元素数：`16 × 32 × 128 = 65,536`
- 所有层 key + value：`65,536 × 2 × 32 = 4,194,304` 个元素
- 字节数：`4,194,304 × 2 = 8 MB`

即每个 block（16 个 token）占用 **8 MB** 显存。

---

## 三、num_gpu_blocks 与 num_cpu_blocks 的计算

`vllm/worker/worker.py` 中的 `profile_num_available_blocks` 分两步完成计算。

**第一步：测出模型峰值显存**

用空 KV Cache（`kv_caches=[(None, None)] * num_layers`）执行一次最大 batch 的前向推理，记录模型权重与激活的峰值显存占用：

```python
peak_memory = torch.cuda.max_memory_allocated()  # bytes
```

**第二步：计算可分配的 block 数**

```python
block_bytes = CacheEngine.get_cache_block_size(block_size, model_config, parallel_config)

num_gpu_blocks = int((total_gpu_memory * gpu_memory_utilization - peak_memory) // block_bytes)
num_cpu_blocks = int(cpu_swap_space_bytes // block_bytes)
```

---

## 关键文件

| 文件 | 相关内容 |
|---|---|
| `vllm/worker/cache_engine.py` | `get_cache_block_size`、`get_key_block_shape`、`get_value_block_shape` |
| `vllm/worker/worker.py` | `profile_num_available_blocks` |
| `vllm/config.py` | `ModelConfig`（head_size/num_heads/num_layers）、`CacheConfig`（block_size） |
