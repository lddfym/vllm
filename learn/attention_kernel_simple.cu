/*
 * 简化版 PagedAttention Kernel（教学用）
 * 对应原版：csrc/attention/attention_kernels.cu
 *
 * 目的：剥去向量化和 thread group 细节，只保留核心的数据分配与归约设计。
 *
 * ── 与原版的主要区别 ────────────────────────────────────────────────────────
 *
 * 1. K cache 内存布局
 *    原版：[num_blocks, num_heads, head_size/x, block_size, x]
 *          x = 16/sizeof(scalar_t)，把 token 维打散到最内层，
 *          目的是让向量化 load（LDG.128）在 head_size 维和 token 维同时对齐
 *    简化：[num_blocks, num_heads, head_size, block_size]  （与 V cache 相同）
 *          每次标量 load 依然 coalesced（同 warp 各 lane 取同一行的相邻列）
 *
 * 2. Thread Group 设计
 *    原版：THREAD_GROUP_SIZE = WARP_SIZE / BLOCK_SIZE（例：2）
 *          一个 thread group 共同计算一个 token 的 Q·K 点积，
 *          每次 load 16 bytes（uint2/uint4），最后 group 内 shfl 归约
 *    简化：去掉 thread group，lane i 独立负责 token i，
 *          标量 load，点积直接在 lane 内循环 HEAD_SIZE 次
 *
 * 3. V 阶段线程分配
 *    原版：lane 同时映射到 (head_dim 行组, token 块)，每个 lane 只需
 *          accs[NUM_ROWS_PER_THREAD]（8 个 float），warp 内归约只需 1 轮
 *    简化：lane → token，每个 lane 维护 accs[HEAD_SIZE]（128 个 float，
 *          寄存器压力大），warp 内对每个 d 做完整 5 轮 XOR butterfly
 *
 * 4. 数据类型
 *    原版：用 Vec<scalar_t,N>::Type / FloatVec 做 fp16 向量化运算
 *    简化：全部转换为 float 做标量运算
 *
 * ── 保留的核心设计（与原版完全一致）────────────────────────────────────────
 *
 * - Grid 与 CUDA block 映射：一个 CUDA block 负责一个 (seq, head) 对
 * - warp → KV page 分配（步长 NUM_WARPS 轮转）
 * - 两级归约：warp 内 XOR butterfly + 跨 warp shared memory 树形归约
 * - Numerically stable softmax（减去 qk_max 再 exp）
 * - Shared memory 两阶段复用：logits[] (阶段一/三) → out_smem[] (阶段五)
 */

#include <float.h>
#define WARP_SIZE 32

namespace vllm_simple {

// ── 工具：block 内两级归约 ─────────────────────────────────────────────────

// 求 block 内所有线程的 float 最大值，结果广播给所有线程
// red_smem 需要至少 NUM_WARPS 个 float
template<int NUM_WARPS>
inline __device__ float block_max(float* red_smem, float val) {
    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;

    // 第一级：warp 内 XOR butterfly（5 轮，mask=16,8,4,2,1）
    // 每轮：lane_i 与 lane_{i^mask} 交换并取 fmax
    // 结束后：warp 内每个 lane 都持有该 warp 的最大值
    for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2)
        val = fmaxf(val, __shfl_xor_sync(~0u, val, mask));
    if (lane == 0) red_smem[warp] = val;
    __syncthreads();

    // 第二级：lane 0..NUM_WARPS-1 读 red_smem，再做 log2(NUM_WARPS) 轮归约
    // __shfl_sync(..., 0) 把结果广播给全 block 所有线程
    val = (lane < NUM_WARPS) ? red_smem[lane] : -FLT_MAX;
    for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2)
        val = fmaxf(val, __shfl_xor_sync(~0u, val, mask));
    return __shfl_sync(~0u, val, 0);
}

// 求 block 内所有线程的 float 总和，结果广播给所有线程
// red_smem 需要至少 NUM_WARPS 个 float
template<int NUM_WARPS>
inline __device__ float block_sum(float* red_smem, float val) {
    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;

    for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2)
        val += __shfl_xor_sync(~0u, val, mask);
    if (lane == 0) red_smem[warp] = val;
    __syncthreads();

    val = (lane < NUM_WARPS) ? red_smem[lane] : 0.f;
    for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2)
        val += __shfl_xor_sync(~0u, val, mask);
    return __shfl_sync(~0u, val, 0);
}

// ── Kernel ────────────────────────────────────────────────────────────────
//
// Grid : (num_heads, num_seqs)   每个 CUDA block 负责一个 (seq, head) 对
// Block: NUM_THREADS 个线程 = NUM_WARPS 个 warp
//
// Shared memory（由 host 分配 max(logits_size, out_smem_size) 字节）：
//   logits_size  = ceil(context_len, BLOCK_SIZE) * BLOCK_SIZE * sizeof(float)
//   out_smem_size = (NUM_WARPS / 2) * HEAD_SIZE * sizeof(float)

template<typename scalar_t, int HEAD_SIZE, int BLOCK_SIZE, int NUM_THREADS>
__global__ void attention_simple(
    scalar_t*       out,           // [num_seqs, num_heads, head_size]
    const scalar_t* q,             // [num_seqs, num_heads, head_size]
    const scalar_t* k_cache,       // [num_blocks, num_heads, head_size, block_size]
    const scalar_t* v_cache,       // [num_blocks, num_heads, head_size, block_size]
    const float     scale,
    const int*      block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int*      context_lens,  // [num_seqs]
    const int       max_num_blocks_per_seq)
{
    constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;

    const int thread_idx = threadIdx.x;
    const int warp_idx   = thread_idx / WARP_SIZE;
    const int lane       = thread_idx % WARP_SIZE;

    const int head_idx  = blockIdx.x;   // 本 CUDA block 负责的 head
    const int seq_idx   = blockIdx.y;   // 本 CUDA block 负责的 seq
    const int num_heads = gridDim.x;

    const int  context_len  = context_lens[seq_idx];
    const int  num_kv_pages = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int* block_table  = block_tables + seq_idx * max_num_blocks_per_seq;

    // ── Shared memory ──────────────────────────────────────────────────────
    // 阶段一/三：logits[token_idx]  存第 i 个 token 的 qk 值（fp32）
    // 阶段五：  复用为 out_smem    跨 warp 归约时存中间结果（fp32）
    extern __shared__ char smem[];
    float* logits = reinterpret_cast<float*>(smem);
    // 前 NUM_WARPS 槽给 qk_max 归约，后 NUM_WARPS 槽给 exp_sum 归约
    __shared__ float red_smem[2 * NUM_WARPS];

    // ══════════════════════════════════════════════════════════════════════
    // 阶段一：Q·K 点积  →  logits[token]（fp32）
    //
    // 数据分配：
    //   CUDA block → 一个 (seq, head) 对
    //   warp_idx   → KV page（步长 NUM_WARPS 轮转所有 page）
    //   lane i     → page 内第 i 个 token（lane >= BLOCK_SIZE 无效）
    //
    // 内存访问（以 BLOCK_SIZE=16 为例）：
    //   q[seq, head, d]           ：所有 lane 读同一地址，L1 broadcast
    //   k_cache[phys, head, d, i] ：lane 0..15 读 [..., d, 0..15]，
    //                               地址连续 → coalesced（16×sizeof(scalar_t) 字节）
    // ══════════════════════════════════════════════════════════════════════
    float qk_max = -FLT_MAX;

    for (int page = warp_idx; page < num_kv_pages; page += NUM_WARPS) {
        const int phys    = block_table[page];
        const int tok_idx = page * BLOCK_SIZE + lane;
        const bool valid  = (lane < BLOCK_SIZE) && (tok_idx < context_len);

        if (valid) {
            const scalar_t* q_ptr = q + seq_idx * num_heads * HEAD_SIZE
                                      + head_idx * HEAD_SIZE;
            // k_cache 布局 [phys, head, d, token]：lane 作为 token 列偏移（步长 1）
            const scalar_t* k_ptr = k_cache + phys * num_heads * HEAD_SIZE * BLOCK_SIZE
                                             + head_idx * HEAD_SIZE * BLOCK_SIZE
                                             + lane;
            float qk = 0.f;
            for (int d = 0; d < HEAD_SIZE; d++)
                qk += (float)q_ptr[d] * (float)k_ptr[d * BLOCK_SIZE];  // 行步长 = BLOCK_SIZE
            qk *= scale;

            logits[tok_idx] = qk;
            qk_max = fmaxf(qk_max, qk);
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // 阶段二：两级归约，求全局 qk_max
    //
    // 第一级（warp 内 XOR butterfly，5 轮，mask=16,8,4,2,1）：
    //   无效 lane 的 qk_max = -FLT_MAX，不影响 fmax 结果
    //   结束后每个 lane 持有该 warp 处理过的所有 token 中的最大 qk
    //   lane 0 写 red_smem[warp_idx]
    //
    // 第二级（借 red_smem，log2(NUM_WARPS) 轮）：
    //   各 lane 读 red_smem，再做一轮 warp 内归约
    //   __shfl_sync(..., 0) 把全局最大值广播给 block 内所有线程
    // ══════════════════════════════════════════════════════════════════════
    qk_max = block_max<NUM_WARPS>(red_smem, qk_max);

    // ══════════════════════════════════════════════════════════════════════
    // 阶段三：Softmax（原地更新 logits[]）
    //
    // 所有线程均分 context_len 个位置（stride = NUM_THREADS）：
    //   val = exp(logits[i] - qk_max)    数值稳定，避免 exp 溢出
    //   logits[i] = val
    //   累加 exp_sum
    // 两级归约得 inv_sum，再遍历一次 logits 完成归一化
    // ══════════════════════════════════════════════════════════════════════
    float exp_sum = 0.f;
    for (int i = thread_idx; i < context_len; i += NUM_THREADS) {
        float val = __expf(logits[i] - qk_max);
        logits[i] = val;
        exp_sum  += val;
    }
    // red_smem 前半段已在 block_max 中使用完毕，后半段存 exp_sum
    exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], exp_sum);

    const float inv_sum = __fdividef(1.f, exp_sum + 1e-6f);
    for (int i = thread_idx; i < context_len; i += NUM_THREADS)
        logits[i] *= inv_sum;
    __syncthreads();  // 确保 logits 全部归一化后阶段四再读

    // ══════════════════════════════════════════════════════════════════════
    // 阶段四：加权 V 累加
    //
    // 数据分配：与阶段一相同（warp → page，lane → token）
    //
    // 每个 lane 维护 HEAD_SIZE 个累加器（寄存器 accs[]）：
    //   accs[d] += logits[tok] * v_cache[page, head, d, lane]
    //
    // 同一 head dim d 的贡献散布在同 warp 的不同 lane（不同 token）
    // → 阶段五需要对每个 d 在 lane 间归约
    //
    // 注意：accs[HEAD_SIZE] 寄存器压力较大（HEAD_SIZE=128 → 512 bytes/thread）
    //       这是简化版相对于原版最主要的效率差距
    // ══════════════════════════════════════════════════════════════════════
    float accs[HEAD_SIZE];
    for (int d = 0; d < HEAD_SIZE; d++) accs[d] = 0.f;

    for (int page = warp_idx; page < num_kv_pages; page += NUM_WARPS) {
        const int phys    = block_table[page];
        const int tok_idx = page * BLOCK_SIZE + lane;
        const bool valid  = (lane < BLOCK_SIZE) && (tok_idx < context_len);

        if (valid) {
            const float w = logits[tok_idx];
            // v_cache 布局 [phys, head, d, token]：与 k_cache 相同
            const scalar_t* v_ptr = v_cache + phys * num_heads * HEAD_SIZE * BLOCK_SIZE
                                             + head_idx * HEAD_SIZE * BLOCK_SIZE
                                             + lane;
            for (int d = 0; d < HEAD_SIZE; d++)
                accs[d] += w * (float)v_ptr[d * BLOCK_SIZE];
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // 阶段五：归约 accs，写最终输出
    //
    // 第一级（warp 内，对每个 d 独立做 XOR butterfly sum，5 轮）：
    //   lane >= BLOCK_SIZE 的无效 lane 其 accs 全 0，不影响求和
    //   结束后每个 lane 的 accs[d] 都等于该 warp 对 d 维的加权和
    //   共 HEAD_SIZE × 5 = 640 条 shfl 指令（原版只需 HEAD_SIZE × 1 轮）
    //
    // 第二级（跨 warp 树形归约，复用 smem，log2(NUM_WARPS) 轮）：
    //   仅 lane 0 参与写/读，每轮：上半 warp 写 out_smem，下半 warp 累加
    //   以 NUM_WARPS=4 为例：
    //     轮1（i=4, mid=2）：warp 2,3（lane 0）写 out_smem；warp 0,1 读并 +=
    //     轮2（i=2, mid=1）：warp 1  （lane 0）写 out_smem；warp 0 读并 +=
    //   warp 0 lane 0 持有全局结果 → 写 out[]（转换为 scalar_t）
    // ══════════════════════════════════════════════════════════════════════

    // 第一级：warp 内对每个 d 归约（XOR butterfly，5 轮）
    for (int d = 0; d < HEAD_SIZE; d++) {
        float acc = accs[d];
        for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2)
            acc += __shfl_xor_sync(~0u, acc, mask);
        accs[d] = acc;
    }

    // 第二级：跨 warp 树形归约
    // logits 已读完，smem 可安全覆盖
    __syncthreads();
    float* out_smem = reinterpret_cast<float*>(smem);

    for (int i = NUM_WARPS; i > 1; i /= 2) {
        const int mid = i / 2;
        // 上半 warp 的 lane 0 把 accs 写入 out_smem
        if (warp_idx >= mid && warp_idx < i && lane == 0) {
            float* dst = out_smem + (warp_idx - mid) * HEAD_SIZE;
            for (int d = 0; d < HEAD_SIZE; d++) dst[d] = accs[d];
        }
        __syncthreads();
        // 下半 warp 的 lane 0 读 out_smem 并累加
        if (warp_idx < mid && lane == 0) {
            const float* src = out_smem + warp_idx * HEAD_SIZE;
            for (int d = 0; d < HEAD_SIZE; d++) accs[d] += src[d];
        }
        __syncthreads();
    }

    // warp 0 lane 0 写最终结果（fp32 → scalar_t）
    if (warp_idx == 0 && lane == 0) {
        scalar_t* out_ptr = out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
        for (int d = 0; d < HEAD_SIZE; d++)
            out_ptr[d] = (scalar_t)accs[d];
    }
}

} // namespace vllm_simple

// ── 与原版的设计对比总结 ──────────────────────────────────────────────────
//
// ┌────────────────────┬──────────────────────────────┬──────────────────────┐
// │ 设计点             │ 简化版                        │ 原版                 │
// ├────────────────────┼──────────────────────────────┼──────────────────────┤
// │ K cache 布局       │ [blk, head, head_size,        │ [blk, head,          │
// │                    │  block_size]                  │  head_size/x,        │
// │                    │                               │  block_size, x]      │
// ├────────────────────┼──────────────────────────────┼──────────────────────┤
// │ Q·K 阶段          │ lane → token（1:1 映射）       │ thread group → token │
// │ 线程分配           │ 每 lane 独立做 HEAD_SIZE 次乘加│ group 内分担 load,   │
// │                    │                               │ 最后 group 内 shfl   │
// ├────────────────────┼──────────────────────────────┼──────────────────────┤
// │ 每次 load 宽度     │ 1 个 scalar（标量）            │ 16 bytes（LDG.128）  │
// ├────────────────────┼──────────────────────────────┼──────────────────────┤
// │ V 阶段线程分配     │ lane → token                  │ lane →               │
// │                    │ accs[HEAD_SIZE]（128 float）   │ (head_dim行, token块)│
// │                    │ 寄存器压力大                  │ accs[8 float]        │
// ├────────────────────┼──────────────────────────────┼──────────────────────┤
// │ V warp 内归约轮数  │ HEAD_SIZE × 5 = 640 条 shfl   │ HEAD_SIZE × 1 = 128  │
// ├────────────────────┼──────────────────────────────┼──────────────────────┤
// │ 两级归约结构       │ 完全相同                      │ 完全相同             │
// │ Shared mem 复用    │ 完全相同                      │ 完全相同             │
// │ Stable softmax     │ 完全相同                      │ 完全相同             │
// └────────────────────┴──────────────────────────────┴──────────────────────┘
//
// 原版 V 阶段的核心优化逻辑（为什么 lane → (head_dim行, token块) 更好）：
//   BLOCK_SIZE=16, fp16 时：V_VEC_SIZE=8, NUM_V_VECS_PER_ROW=2
//   lane 0 负责 (row 0, token 0..7)，lane 1 负责 (row 0, token 8..15)
//   → 同一 row 只有 2 个 lane，warp 内只需 1 轮 shfl（mask=1）即可合并
//   → 每个 lane 的 accs 只需 NUM_ROWS_PER_THREAD=8 个 float（寄存器压力低 16×）
