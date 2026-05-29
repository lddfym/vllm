# vLLM Cache Ops：cache_kernels.cu 实现原理

## 一、KV Cache 内存布局

KV cache tensor 的第一维是 block 索引，以 key cache 为例：

```
[num_gpu_blocks, num_heads, head_size/x, block_size, x]
```

PyTorch 默认 C-contiguous 存储（行主序），第一维最外层，因此 block k 的全部数据
（该层所有 head、所有 token 位置）在物理内存上连续排列：

```
地址 →
┌────────────┬────────────┬────────────┐
│  Block 0   │  Block 1   │  Block 2…  │
└────────────┴────────────┴────────────┘
 data_ptr     +block_bytes  +2×block_bytes
```

`block_size_in_bytes = src.element_size() * src[0].numel()`：
- `src[0]` 沿第一维取 block 0，得到 `[num_heads, head_size/x, block_size, x]`
- `.numel()` = 单个 block 的元素总数（与内层维度结构无关）

这一布局使得拷贝时无需关心内层维度，直接以 `void*` + 字节偏移寻址即可，
是后续所有四个操作的共同前提。

---

## 二、swap_blocks：CPU↔GPU 块搬运

`swap_blocks` 在 CPU 内存（CPU 侧 KV cache）和 GPU 显存（GPU 侧 KV cache）之间移动 KV block，
实现方式是在 CPU 侧循环发出 `cudaMemcpyAsync`，**不使用 CUDA kernel**。

### 拷贝实现（`csrc/cache_kernels.cu` L9–47）

```cpp
for (const auto& pair : block_mapping) {
    cudaMemcpyAsync(
        dst_ptr + dst_block_number * block_size_in_bytes,
        src_ptr + src_block_number * block_size_in_bytes,
        block_size_in_bytes,
        memcpy_type,   // HostToDevice / DeviceToHost / DeviceToDevice
        stream);
}
```

`memcpy_type` 由 `src_device`/`dst_device` 在运行时决定，同一函数覆盖 swap in / swap out 两个方向。

### 为什么可以真正异步

`swap_blocks` 内部通过 `at::cuda::getCurrentCUDAStream()` 获取当前 CUDA stream，
Python 层调用前已执行：

```python
# vllm/worker/cache_engine.py  _swap()
with torch.cuda.stream(self.cache_stream):
    cache_ops.swap_blocks(...)
```

`with torch.cuda.stream()` 将 PyTorch 当前 stream 切换为 `cache_stream`，
因此 `cudaMemcpyAsync` 投入的是专用 `cache_stream` 而非 compute stream。

异步生效的前提是 CPU 端内存必须为 **pinned memory**（页锁定）——
CPU cache 分配时使用 `pin_memory=True` 正是出于此目的（详见 `kv_cache_optimizations.md` 第三节）。
两者共同保证 `cudaMemcpyAsync` 调用后立即返回，DMA 在后台由 `cache_stream` 驱动完成。

→ `cache_stream` 与 compute stream 的并行协作机制详见 `kv_cache_optimizations.md` 第二节。

---

## 三、CUDA Kernel 基础

CUDA kernel 是运行在 GPU 上的函数，被大量线程并行执行。线程按三级组织：

```
Grid（网格）
  └─ Thread Block（线程块）× (gridDim.x × gridDim.y × gridDim.z) 个
       blockIdx.x ∈ [0, gridDim.x)，blockIdx.y ∈ [0, gridDim.y)，blockIdx.z ∈ [0, gridDim.z)
       ├─ [硬件] Warp（线程束）× ⌈(blockDim.x × blockDim.y × blockDim.z) / 32⌉ 个，每 Warp 固定 32 线程
       └─ Thread（线程）× (blockDim.x × blockDim.y × blockDim.z) 个
            threadIdx.x ∈ [0, blockDim.x)，threadIdx.y ∈ [0, blockDim.y)，threadIdx.z ∈ [0, blockDim.z)
```

**Warp** 是 GPU 硬件的调度单元：硬件将 Block 内线程按 x→y→z 顺序线性化后，每 32 个连续编号打包成一个 Warp，以 SIMT（Single Instruction, Multiple Threads）方式执行——同一 Warp 内所有线程在同一时钟周期执行同一条指令，但各自使用不同的寄存器和数据。Warp 没有自己的维度参数，程序员通过 `threadIdx` 访问的是 Block 内全量线程（`blockDim(1024)` 时 `threadIdx.x` 范围是 0–1023）。Block 内线程数不是 32 的倍数时，最后一个 Warp 用空闲通道补齐（padding），不影响正确性但浪费执行槽位。

Launch 时通过 `<<<grid, block>>>` 指定两级维度，每级维度均为三维：

- `dim3 grid(x, y, z)`：Grid 中 Thread Block 的数量，z 默认为 1（kernel 内读作 `gridDim.x/y/z`）
- `dim3 block(x, y, z)`：每个 Thread Block 内线程的数量，z 默认为 1（kernel 内读作 `blockDim.x/y/z`）

### 为何有三个方向

三个方向不对应任何物理硬件结构——GPU 调度器将所有 Thread Block 视为等价的工作单元，
以任意顺序分发给 SM（流式多处理器）执行，x/y/z 之间没有优先级或执行顺序的区别。

三维的意义在于**为程序员提供天然的多维索引空间**，避免手动做下标转换。

### Host / Device 模型

CUDA 程序分两侧：
- **Device（GPU）**：kernel 函数，用 `__global__` 修饰，由大量 GPU 线程并行执行
- **Host（CPU）**：普通 C++ 函数，负责分配显存、传输数据、决定启动多少线程

Host 代码通过 `kernel<<<grid, block>>>(args)` 语法将任务投入 GPU。
**这一调用在 CPU 上发出，kernel 本体在 GPU 上执行**；调用本身异步，CPU 投递完立即返回，
GPU 在后台并行执行所有线程。

### Kernel 内置变量

| 变量 | 含义 |
|---|---|
| `threadIdx.x/y/z` | 线程在 Thread Block 内的下标 |
| `blockIdx.x/y/z` | 当前 Thread Block 在 Grid 中的下标 |
| `blockDim.x/y/z` | Thread Block 的维度大小（= launch 时 `block` 参数） |

**等价写法**：`dim3(1)` 等同于 `dim3(1, 1, 1)`——未指定的维度默认为 1，
因此 `block(1)`、`block(1,1)`、`block(1,1,1)` 三者完全等价。

### 示例：对 3D 数组中每个元素乘以 2

数据形状 `float data[D][H][W]`，将 grid 设计为 `(D, H, W)` 三维，
配套设置 `block(1)`（即每个 Thread Block 只含 1 个线程，该线程独立处理一个 `(d,h,w)` 位置），
则每个 Thread Block 的 `blockIdx` 直接对应三维坐标，无需手动线性化：

```cpp
// ---- Device（GPU 线程执行）----
__global__ void scale_3d(float* data, int H, int W) {
    int d = blockIdx.x;
    int h = blockIdx.y;
    int w = blockIdx.z;
    data[d * H * W + h * W + w] *= 2.f;
}

// ---- Host：完整调用流程 ----
int D = 4, H = 32, W = 32;

// 1. 在 CPU 内存中准备原始数据（h_data 是普通堆内存）
float* h_data = new float[D * H * W];
for (int i = 0; i < D * H * W; i++) h_data[i] = (float)i;

// 2. 在 GPU 显存中分配空间，并将数据从 CPU 拷贝到 GPU
float* d_data;
cudaMalloc(&d_data, D * H * W * sizeof(float));
cudaMemcpy(d_data, h_data, D * H * W * sizeof(float), cudaMemcpyHostToDevice);

// 3. 启动 kernel（CPU 发出调用，GPU 异步执行）
//    grid 三维与数据形状对应，每个 Thread Block 处理一个 (d,h,w) 位置
dim3 grid(D, H, W);   // D×H×W 个 Thread Block
dim3 block(1);         // 每个 Thread Block 1 个线程
scale_3d<<<grid, block>>>(d_data, H, W);  // CPU 立即返回，GPU 开始执行

// 4. 等待 GPU 完成，将结果从 GPU 拷贝回 CPU
cudaDeviceSynchronize();  // 阻塞 CPU，直到 GPU 完成
cudaMemcpy(h_data, d_data, D * H * W * sizeof(float), cudaMemcpyDeviceToHost);

// 5. 释放 GPU 显存
cudaFree(d_data);
```

### 跨步循环（stride loop）

当元素数超过线程数时，单个线程以步长 `blockDim.x` 循环处理多个元素：

```cpp
__global__ void kernel(float* data, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        data[i] = ...;
    }
}
// Launch: kernel<<<1, 1024>>>(data, n);
// Thread 0 处理 0, 1024, 2048, …
// Thread 1 处理 1, 1025, 2049, …
```

所有线程的访问地址连续分布，符合 GPU 的合并访问（coalesced access）要求。

---

## 四、copy_blocks：GPU 内批量复制

### block_mapping 结构

`copy_blocks` 的 src 和 dst 均在 GPU 上。调用者传入
`block_mapping: map<int64_t, vector<int64_t>>`：
- key（`int64_t`）：源 block 编号（src_block）
- value（`vector<int64_t>`）：该源 block 要复制到的所有目标 block 编号列表

这个 map 展开后得到若干 `(src_block, dst_block)` 对，
每一对称为一个 **pair**，`num_pairs` 是展开后的总对数。例如：

```
block_mapping = { 5: [10, 11, 12],  7: [20] }
→ pairs: (5,10), (5,11), (5,12), (7,20)   →  num_pairs = 4
```

支持一对多的设计用于 beam search 或 prefix caching 的写时复制（Copy-on-Write）：
一个已缓存的 prefix block 可以一次性复制到多个目标 block，各序列后续独立写入自己的副本。
若用 `cudaMemcpyAsync` 循环，需发出 `2 × num_layers × num_pairs` 次调用；
改用单次 kernel launch 可将所有层、所有 pair 合并处理。

### Host 侧：copy_blocks 完整源码与逐行解释

```cpp
// csrc/cache_kernels.cu  L82–138
void copy_blocks(
  std::vector<torch::Tensor>& key_caches,    // 所有层的 key cache tensor（GPU）
  std::vector<torch::Tensor>& value_caches,  // 所有层的 value cache tensor（GPU）
  const std::map<int64_t, std::vector<int64_t>>& block_mapping) {

  int num_layers = key_caches.size();
  if (num_layers == 0) { return; }

  // ---- 步骤 1：将各层 cache 的 GPU 指针收集到数组 ----
  // kernel 按 layer_idx 索引取对应层的 GPU 地址；
  // 这里将指针强转为 int64_t 存储，之后搬到 GPU 供 kernel 使用
  int64_t key_cache_ptrs[num_layers];
  int64_t value_cache_ptrs[num_layers];
  for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
    key_cache_ptrs[layer_idx]   = reinterpret_cast<int64_t>(key_caches[layer_idx].data_ptr());
    value_cache_ptrs[layer_idx] = reinterpret_cast<int64_t>(value_caches[layer_idx].data_ptr());
  }

  // ---- 步骤 2：将 block_mapping 展开为 [src0,dst0, src1,dst1, ...] 数组 ----
  // kernel 内通过 block_mapping[2*pair_idx] / [2*pair_idx+1] 取 src/dst，
  // 因此先将 map<int64_t, vector<int64_t>> 展开为连续的 int 数组
  std::vector<int> block_mapping_vec;
  for (const auto& pair : block_mapping) {
    int src_block_number = pair.first;
    for (int dst_block_number : pair.second) {
      block_mapping_vec.push_back(src_block_number);
      block_mapping_vec.push_back(dst_block_number);
    }
  }
  int num_pairs = block_mapping_vec.size() / 2;

  // ---- 步骤 3：将辅助数据搬到 GPU 显存 ----
  // kernel 只能访问 GPU 显存；from_blob 包装 CPU 数组，.to(cache_device) 触发 CPU→GPU 拷贝。
  // NOTE: .to(cache_device) 会隐式同步 CPU 和 GPU
  torch::Tensor key_cache_ptrs_tensor =
      torch::from_blob(key_cache_ptrs, {num_layers}, torch::kInt64).to(cache_device);
  torch::Tensor value_cache_ptrs_tensor =
      torch::from_blob(value_cache_ptrs, {num_layers}, torch::kInt64).to(cache_device);
  torch::Tensor block_mapping_tensor =
      torch::from_blob(block_mapping_vec.data(), {2 * num_pairs}, torch::kInt).to(cache_device);

  // ---- 步骤 4：启动 kernel ----
  const int numel_per_block = key_caches[0][0].numel();  // 单个 block 的元素数（单层 key 或 value）
  // grid 第 0 维 = num_layers，第 1 维 = num_pairs
  // → kernel 内 blockIdx.x 取值 0…num_layers-1（层索引），blockIdx.y 取值 0…num_pairs-1（pair 索引）
  dim3 grid(num_layers, num_pairs);
  dim3 block(std::min(1024, numel_per_block));  // 每个 Thread Block 的线程数，上限 1024
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  // AT_DISPATCH_FLOATING_TYPES_AND2 根据 tensor dtype（fp16/bf16/fp32）实例化模板，
  // scalar_t 即该 dtype 对应的 C++ 类型
  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half, at::ScalarType::BFloat16,
    key_caches[0].scalar_type(), "copy_blocks_kernel", ([&] {
      vllm::copy_blocks_kernel<scalar_t><<<grid, block, 0, stream>>>(
          key_cache_ptrs_tensor.data_ptr<int64_t>(),
          value_cache_ptrs_tensor.data_ptr<int64_t>(),
          block_mapping_tensor.data_ptr<int>(),
          numel_per_block);
    }));
}
```

### Device 侧：kernel 源码与逐行解释

```cpp
// csrc/cache_kernels.cu  L52–78
template<typename scalar_t>
__global__ void copy_blocks_kernel(
  int64_t* key_cache_ptrs,               // 各层 key cache 的 GPU 指针数组
  int64_t* value_cache_ptrs,             // 各层 value cache 的 GPU 指针数组
  const int* __restrict__ block_mapping, // 展开后的 pair 数组 [src0,dst0, src1,dst1, ...]
  const int numel_per_block) {           // 单个 block 的元素总数

  // grid.x=num_layers → blockIdx.x 取值 0…num_layers-1，即当前 Thread Block 负责的层
  const int layer_idx = blockIdx.x;
  // grid.y=num_pairs → blockIdx.y 取值 0…num_pairs-1，即当前 Thread Block 负责的 (src,dst) pair
  const int pair_idx  = blockIdx.y;

  // 按 layer_idx 取本层 key/value cache 的 GPU 起始地址
  scalar_t* key_cache   = reinterpret_cast<scalar_t*>(key_cache_ptrs[layer_idx]);
  scalar_t* value_cache = reinterpret_cast<scalar_t*>(value_cache_ptrs[layer_idx]);

  // 按 pair_idx 取 src/dst block 编号（数组中每 2 个元素为一对）
  int src_block_number = block_mapping[2 * pair_idx];
  int dst_block_number = block_mapping[2 * pair_idx + 1];

  // 计算 src/dst block 在 cache 数组中的起始元素偏移
  const int src_block_offset = src_block_number * numel_per_block;
  const int dst_block_offset = dst_block_number * numel_per_block;

  // 跨步循环（见第三节）：Thread 0 处理 i=0, blockDim.x, 2×blockDim.x, …
  for (int i = threadIdx.x; i < numel_per_block; i += blockDim.x) {
    key_cache[dst_block_offset + i] = key_cache[src_block_offset + i];
  }
  for (int i = threadIdx.x; i < numel_per_block; i += blockDim.x) {
    value_cache[dst_block_offset + i] = value_cache[src_block_offset + i];
  }
}
```

key 和 value 分两轮循环：同一 Thread Block 内所有线程先完成 key 拷贝，再进行 value 拷贝。

### stream 与同步

`copy_blocks` 使用 `at::cuda::getCurrentCUDAStream()`，调用方不切换 stream，
运行在 compute stream 上。步骤 3 中 `.to(cache_device)` 会隐式同步 CPU 和 GPU，
因此通常在每个 step 开始前调用，不与 swap 并行。

---

## 五、reshape_and_cache：前向 KV 写入 Cache

第二、四节（swap_blocks / copy_blocks）描述的是 cache 已有内容的搬运。`reshape_and_cache` 是另一条路径：
每个 decode step 完成 Attention 前向计算后，将本批 token 新生成的 key/value 向量写入 cache，
同时完成布局转换（标准线性布局 → 分块向量化布局）。

**调用时机**：KV cache（`key_caches` / `value_caches`）在初始化时已整块预分配完毕，每层一个 tensor。
每次前向传播中，每个 Transformer 层的 Attention 计算结束后立即调用 `reshape_and_cache`，
将本层本 batch 新计算出的 key/value 写入该层的预分配 cache 对应位置，随后继续下一层计算。
此处的 `num_tokens` 是当前 batch 中的 token 总数：
decode 阶段每个 sequence 各贡献 1 个新 token（`num_tokens = batch_size`）；
prefill 阶段则是所有 sequence 的 prompt token 之和。

**为何用 CUDA kernel 而非 PyTorch API**：核心挑战是 `slot_mapping` 导致的 **scattered write**
——每个 token 需要写入 cache 中不同的非连续位置（不同的 `block_idx` / `block_offset`）。
理论上可以用 PyTorch 的 `index_copy`/`scatter` + `view`/`permute` 实现，
但这需要至少两步操作（reshape 一次 kernel、scatter 写入一次 kernel）并产生中间临时 tensor；
此外 `[..., head_size/x, block_size, x]` 的 x 维拆分不是 PyTorch 标准 API 能直接产生的。
自定义 CUDA kernel 将布局转换与 scattered write 融合为单次 pass，无中间分配，单次 launch 覆盖所有 token。

### slot_mapping：token 到 cache 位置的映射

调度器为每个 token 分配一个全局 slot 编号：

```
slot = block_idx * block_size + block_offset
```

`slot_mapping[token_idx]` 即为第 `token_idx` 个 token 对应的 slot。
kernel 从 slot 还原出 block 内的位置：

```cpp
const int slot_idx    = slot_mapping[token_idx];
const int block_idx   = slot_idx / block_size;   // 该 token 属于哪个 block
const int block_offset = slot_idx % block_size;  // 在 block 内的偏移（第几个 token 位置）
```

### grid/block 设计

```cpp
// Host 侧（csrc/cache_kernels.cu L202–203）
dim3 grid(num_tokens);                              // 每个 Thread Block 负责一个 token
dim3 block(std::min(num_heads * head_size, 512));   // 线程数 = min(每 token 的元素数, 512)
```

`grid(num_tokens)` 使 `blockIdx.x` 直接对应 token 索引：

```cpp
const int token_idx = blockIdx.x;
```

每个 Thread Block 负责将 `token_idx` 这个 token 的所有 head 的 key/value 写入 cache，
内部使用跨步循环遍历 `num_heads * head_size` 个元素。

### key 的布局转换

前向计算输出的 key 布局为标准线性格式：

```
key: [num_tokens, num_heads, head_size]
     第 token_idx 个 token、第 head_idx 个 head、第 head_offset 个维度
     → 线性下标 i = head_idx * head_size + head_offset
```

cache 中 key 的目标布局为：

```
key_cache: [num_blocks, num_heads, head_size/x, block_size, x]
```

其中 `x = 16 / element_size`（fp16 → x=8，fp32 → x=4），将 `head_size` 维拆分为
`(head_size/x)` 组、每组 `x` 个连续元素，使 GPU 可以用一条 16-byte 向量化指令读取 x 个值。

转换逻辑：

```cpp
const int head_idx    = i / head_size;    // 第几个 head
const int head_offset = i % head_size;    // head 内的第几个维度元素

// 将 head_offset 拆成 (x_idx, x_offset)
const int x_idx    = head_offset / x;    // 第几组（head_size/x 维的下标）
const int x_offset = head_offset % x;   // 组内第几个元素（x 维的下标）

// 目标下标：依次对应 [block_idx][head_idx][x_idx][block_offset][x_offset]
const int tgt_key_idx =
    block_idx    * num_heads * (head_size / x) * block_size * x
  + head_idx     * (head_size / x) * block_size * x
  + x_idx        * block_size * x
  + block_offset * x
  + x_offset;
```

### value 的布局转换

value 的目标布局为：

```
value_cache: [num_blocks, num_heads, head_size, block_size]
```

无需 x 维拆分，只需将 `head_offset` 和 `block_offset` 的顺序对调：

```cpp
const int tgt_value_idx =
    block_idx    * num_heads * head_size * block_size
  + head_idx     * head_size * block_size
  + head_offset  * block_size
  + block_offset;
```

### 两种布局的设计动机

key cache 和 value cache 使用不同布局，是为了匹配 Attention 计算时各自的读取方式：

- **key**：自回归生成时，当前 step 的 query 需要与上下文中所有历史 token 的 key 计算点积
  （即 `Q_new · K[0..t-1]^T`），因此需要读取大量历史 key 向量。
  `[..., head_size/x, block_size, x]` 的布局将每个 key 向量的 head 维度
  按 x 个元素为一组分段（`x_idx = head_offset / x`），
  确保同一 token（`block_offset`）在同一分段内的 x 个连续 head 元素在显存中物理相邻
  （最内层下标 `x_offset = 0..x-1`），
  GPU 可用一条 16-byte 向量化指令（`float4` / `half8`）一次读取这 x 个元素，
  提高 QK 点积的内存带宽利用率。

- **value**：计算加权和 `sum(attn_weight[i] * V[i, :])` 时，对于每个 head 维度位置 `head_offset`，
  需要将 block 内所有 token 在该位置的 value 归约。
  `[..., head_size, block_size]` 的布局将同一 `head_offset` 下所有 token 的 value 连续存储，
  天然契合按 head_offset 顺序归约的访问模式，无需额外的 x 维向量化。

### 完整 kernel 源码

```cpp
// csrc/cache_kernels.cu  L142–182
template<typename scalar_t>
__global__ void reshape_and_cache_kernel(
  const scalar_t* __restrict__ key,      // [num_tokens, num_heads, head_size]  输入
  const scalar_t* __restrict__ value,    // [num_tokens, num_heads, head_size]  输入
  scalar_t* __restrict__ key_cache,      // [num_blocks, num_heads, head_size/x, block_size, x]  输出
  scalar_t* __restrict__ value_cache,    // [num_blocks, num_heads, head_size, block_size]  输出
  const int* __restrict__ slot_mapping,  // [num_tokens]  token→slot 映射
  const int key_stride,    // = key.stride(0)，单位是元素数（非字节）；连续 tensor 下 = num_heads × head_size
  const int value_stride,  // 同 key_stride；连续 tensor 下 = num_heads × head_size
  const int num_heads,
  const int head_size,
  const int block_size,
  const int x) {

  // blockIdx.x = token 索引（grid.x = num_tokens）
  const int token_idx    = blockIdx.x;
  const int slot_idx     = slot_mapping[token_idx];
  const int block_idx    = slot_idx / block_size;
  const int block_offset = slot_idx % block_size;

  // 跨步循环：遍历该 token 的所有 num_heads * head_size 个元素
  const int n = num_heads * head_size;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    // __ldg：通过 L1 只读数据缓存（L1 Read-Only Data Cache）读取，适合 const __restrict__ 只读数组
    const int src_key_idx   = token_idx * key_stride   + i;
    const int src_value_idx = token_idx * value_stride + i;

    const int head_idx    = i / head_size;
    const int head_offset = i % head_size;
    const int x_idx    = head_offset / x;
    const int x_offset = head_offset % x;

    // key：写入 [block_idx][head_idx][x_idx][block_offset][x_offset]
    const int tgt_key_idx =
        block_idx    * num_heads * (head_size / x) * block_size * x
      + head_idx     * (head_size / x) * block_size * x
      + x_idx        * block_size * x
      + block_offset * x
      + x_offset;
    // value：写入 [block_idx][head_idx][head_offset][block_offset]
    const int tgt_value_idx =
        block_idx   * num_heads * head_size * block_size
      + head_idx    * head_size * block_size
      + head_offset * block_size
      + block_offset;

    key_cache[tgt_key_idx]     = __ldg(&key[src_key_idx]);
    value_cache[tgt_value_idx] = __ldg(&value[src_value_idx]);
  }
}
```

---

## 六、gather_cached_kv：从 Cache 读回 KV

`gather_cached_kv` 是 `reshape_and_cache` 的逆操作：将已缓存的 KV 从分块向量化布局
读回为标准线性格式 `[num_tokens, num_heads, head_size]`，用于 Attention 计算前的数据准备。

### 与 reshape_and_cache 的对应关系

grid/block 设计、slot_mapping 用法和维度计算与第五节完全一致；唯一区别是 **src/dst 互换**：

| | reshape_and_cache | gather_cached_kv |
|---|---|---|
| 输入 | `key/value`（标准线性） | `key_cache/value_cache`（分块向量化） |
| 输出 | `key_cache/value_cache` | `key/value`（标准线性） |
| 下标计算 | tgt = cache 布局 | src = cache 布局 |

### 优化版 kernel：loop unrolling

实际调用的是 `gather_cached_kv_kernel_optimized`（L272–344），
通过 `unroll_factor = 4` 将每次迭代处理的元素数提升 4 倍。

在阅读代码前，先了解其中用到的两个 CUDA 特性：

**`__ldg(ptr)`**：指示 GPU 通过 **L1 只读数据缓存（L1 Read-Only Data Cache）** 读取
`ptr` 所指向的数据，而不走普通 load 指令使用的 **L1 数据缓存（L1 Data Cache，可读写）**。
两者的区别在于缓存失效（cache invalidation）的处理方式：
L1 Data Cache 需要支持读写一致性——当某个线程对某地址执行 store 时，
L1 Data Cache 中缓存了该地址的缓存行会被标记为无效并让出空间；
L1 Read-Only Data Cache 因为数据被声明为只读，无需监听 store 操作，
不存在缓存失效问题，有效缓存容量更稳定，适合 `const __restrict__` 修饰的只读数组
（如此处的 `key_cache`、`value_cache`）。

**`#pragma unroll`**：编译器指令，将紧随其后的循环**展开**（unroll）：
原本的循环控制（计数器更新、条件跳转）被消除，循环体直接复制 N 份变为顺序代码。
展开后编译器能看到多条相邻的 load 指令（从显存读数据到寄存器）和 store 指令
（从寄存器写数据到显存），从而将它们交错排列，让多条 load 的显存等待时间相互覆盖，
提升指令级并行（ILP）。

```cpp
// csrc/cache_kernels.cu  L272–344
template <typename scalar_t>
__global__ void gather_cached_kv_kernel_optimized(
    scalar_t* key, scalar_t* value,
    const scalar_t* key_cache, const scalar_t* value_cache,
    const int* slot_mapping,
    const int key_stride, const int value_stride,
    const int num_heads, const int head_size,
    const int block_size, const int x) {

  // slot_mapping → block_idx / block_offset（同第五节）
  const int token_idx    = blockIdx.x;
  const int slot_idx     = slot_mapping[token_idx];
  const int block_idx    = slot_idx / block_size;
  const int block_offset = slot_idx % block_size;

  const int dim = num_heads * head_size;
  assert(dim % 4 == 0);            // 已知约束，保证整除
  const int unroll_factor = 4;
  const int unrolled_dim  = dim / unroll_factor;  // 每个线程的迭代次数

  for (int i = threadIdx.x; i < unrolled_dim; i += blockDim.x) {
    int tgt_key_indices[4], tgt_value_indices[4];
    int src_key_indices[4], src_value_indices[4];
    scalar_t keys[4], values[4];

    // ---- 阶段 1：计算 4 个位置的下标，批量 __ldg 读取 ----
    // 4 个元素均匀分布在 [0, dim) 区间：
    //   i=0 → index = 0, unrolled_dim, 2*unrolled_dim, 3*unrolled_dim
    //   i=1 → index = 1, 1+unrolled_dim, ...
    #pragma unroll
    for (int j = 0; j < unroll_factor; ++j) {
      int index = i + j * unrolled_dim;

      const int head_idx    = index / head_size;
      const int head_offset = index % head_size;
      const int x_idx       = head_offset / x;
      const int x_offset    = head_offset % x;

      // src = cache 布局（与 reshape_and_cache 的 tgt 公式相同）
      src_key_indices[j] = block_idx * num_heads * (head_size / x) * block_size * x
                           + head_idx * (head_size / x) * block_size * x
                           + x_idx   * block_size * x
                           + block_offset * x + x_offset;
      src_value_indices[j] = block_idx * num_heads * head_size * block_size
                             + head_idx   * head_size * block_size
                             + head_offset * block_size + block_offset;

      tgt_key_indices[j]   = token_idx * key_stride   + index;
      tgt_value_indices[j] = token_idx * value_stride + index;

      keys[j]   = __ldg(&key_cache[src_key_indices[j]]);
      values[j] = __ldg(&value_cache[src_value_indices[j]]);
    }

    // ---- 阶段 2：批量写回 ----
    #pragma unroll
    for (int j = 0; j < unroll_factor; ++j) {
      key[tgt_key_indices[j]]   = keys[j];
      value[tgt_value_indices[j]] = values[j];
    }
  }
}
```

**关键优化**：将索引计算 + 读取（阶段 1）与写回（阶段 2）拆成两个独立的 `#pragma unroll` 循环。
编译器可以将 4 次 `__ldg` 流水线化（在等待内存返回时执行其他指令），再统一执行 4 次连续写入，
减少 load-use 依赖造成的流水线阻塞。

---

## 关键文件

| 文件 | 相关内容 |
|---|---|
| `csrc/cache_kernels.cu` | `swap_blocks`（L9–47）、`copy_blocks_kernel`（L52–78）、`copy_blocks`（L82–138）、`reshape_and_cache_kernel`（L142–182）、`reshape_and_cache`（L186–224）、`gather_cached_kv_kernel`（L228–270）、`gather_cached_kv_kernel_optimized`（L272–344）、`gather_cached_kv`（L348–386） |
| `csrc/cache.cpp` | pybind11 注册 |
| `vllm/worker/cache_engine.py` | `_swap()`、`allocate_cpu_cache()`（pin_memory）、`cache_stream` |
| `learn/kv_cache_optimizations.md` | 第二节（cache_stream/Event）、第三节（pin memory） |
