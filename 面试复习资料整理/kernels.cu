// 1. 向上取整
 #define CEIL(a, b) ((a + b - 1) / (b))
 // 2. FLOAT4，用于向量化访存，以下两种都可以
 // c写法
 #define FLOAT4(value) *(float4*)(&(value))
 // c++写法
 #define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])

// add
int block_size = 1024;
int grid_size = CEIL(CEIL(N,4), block_size);

__global__ void elementwise_add(float* a, float* b, float* c, int N){
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    if(i < N){
        //进行向量化访存
        float4 a4 = FLOAT4(a[i]);
        float4 b4 = FLOAT4(b[i]);
        float4 c4;

        //进行向量化计算，每个线程计算4个元素
        c4.x = a4.x + b4.x;
        c4.y = a4.y + b4.y;
        c4.z = a4.z + b4.z;
        c4.w = a4.w + b4.w;
        FLOAT4(c[i]) = c4;

    }
}

//reduce
int block_size = 1024;
int grid_size = CEIL(N, block_size);

__global__ void reduce(float* input, float* out, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N){
        atomicAdd(out, input[idx]);
    }
}

reduce<<<grid_size, block_size>>>(input, out, N);

// warp shuffle reduce
// 使用warp的优势：每个warp内线程可以并行计算，每个warp之间的线程可以并行计算
// 什么是warp：warp是一个线程组，每个warp内线程可以并行计算，每个warp之间的线程可以并行计算
dim3 block_size(BLOCK_SIZE);
dim3 grid_size(CEIL(N, BLOCK_SIZE));

__global__ void recude_v3(float* input, float* output, int N){
    int warpSize = 4;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int warp_id = threadIdx.x / 4;
    int lane_id = threadIdx.x % 4;

    __shared__ float smem[BLOCK_SIZE];

    float val = (idx < N) ? input[idx] : 0.0f;

    #pragma unroll
    for (int offset = warpSize>>1; offset>0; offset>>=1){
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    if (lane_id == 0){
        smem[warp_id] = val;
    }

    if (warp_id == 0){
        int warp_num = blockDim.x / warpSize;
        val = (lane_id < warp_num) ? smem[lane_id] : 0.0f;
        for (int offset = warpSize >> 1; offset>0; offset>>=1){
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        if (lane_id==0) atomicAdd(output, val);
    }
}

recude_v3<<<grid_size, block_size>>>(input, output, N);

// softmax
// cpu version

void softmax(float* input, float* output, int N){
    int M = *(std::max_element(input, input + N));
    float div = 0;
    for (int i = 0; i < N; i++){
        output[i] = exp(input[i] - M);
        div += output[i];
    }
    for (int i = 0; i < N; i++){
        output[i] /= div;
    }
}

// cuda version
dim3 block_size(BLOCK_SIZE);
dim3 grid_size(CEIL(N, BLOCK_SIZE));

__device__ static float atomicMax(float* adderss, float val){
    int* address_as_i = (int*) adderss;
    int old = *address_as_i;
    int assumed;
    do{
        assumed = old;
        old = atomicCAS(address_as_i, assumed, __float2int(val));
    }while(old != assumed);
    return __int2float(old);
}

__global__ void max_kernel(float* input, float* output, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warps_sieze = 4;
    int wapr_id = threadIdx.x / warp_size;
    int lane_id = threadIdx.x * warp_size;

    __shared__ float smem[BLOCK_SIZE];

    float val = (idx < N) ? input[idx] : 0.0f;

    for (int offset = warp_id>>1; offset>0; offset>>=1){
        val = fmax(val, __shfl_down_sync(0xFFFFFFFF, val, offset)); //
    }

    if (lane_id==0){
        smem[warp_id] = val;
    }
    __syncthreads();

    if(warp_id ==0 ){
        int warp_num = blockDim.x / warp_size;
        val = (lane_id < warp_num) ? smem[lane_id] : 0.0f;
        for (int offset = warp_size >> 1; offset>0; offset>>=1){
            val = fmax(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
        }
        if (lane_id==0) atomicMax(output, val);
    }
}

__global__ void sum_kernel(float* input, float* output, float* max_val, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int wapr_id = threadIdx.x / warp_size;
    int lane_id = threadIdx.x * warp_size;

    __shared__ float smem[BLOCK_SIZE];

    float val = (idx < N) ? expf(input[idx] - *max_val) : 0.0f;
    for(int offset=warp_id>>1; offset>0; offset>>=1){
        val += __shfl_down_sync(0xFFFFFFFF, val, output);
    }
    
    if (lane_id==0) smem[warp_id] = val;
    __syncthreads();

    for(warp_id==0){
        int warp_num= blockDim.x / warp_size;
        val = (lane_id < warp_num) ? smem[lane_id]:0.0f;
        for(int offset=warp_id>>1; offset>0; offset>>=1){
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        if(lane_id==0) atomicAdd(output, val);
    }
}

__global__ void soft_max(float* input, float* output, float* sum, float* max_val, int N){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < N){
        output[idx] = expf(input[idx] - *max_val) / (*sum);
    }
}

int block_size = 256;
int grid_size = CEIL(N, block_size);
max_kernel<<grid_size, block_size>>(input, max_val, N);
sum_kernel<<grid_size, block_size>>(input, sum, N);
soft_max<<grid_size, block_size>>(input, output, sum, max_val, N);

//transpose
 __global__ void transpose(float* input, float* output, int M, int N){

    //input的索引
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row <M && col <N){
        output[col*M + row] = input[row*N + col];
    }
 }

 __global__ void transpose_v2(float* input, float* output, int M, int N){
    //output的索引
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < N && col << M){
        output[col*N + row] = __ldg(input[row*M + col]);
    }
    //__ldg的作用是将input[row*M + col]的值加载到缓存中，避免内存访问
 }

 // sgemm
 __global__ void sgemm(float* A, float* B, float* output, int M, int N, int K){
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    if(row >= M || col>=N) return;

    float accsum = 0.0f;
    for (int i=0; i<K; i++){
        accsum += A[row*K + i] * B[i*N + col];
    }
    output[row*N + col] = accsum;
 }

 // block tile version
 __global__ void sgemm_v2(float* A, float* B, float* output, int M, int N, int K){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if (idx >=M || idy >= N) return;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    const int BM = BLOCK_SIZE;
    const int BN = BLOCK_SIZE;
    const int BK = BLOCK_SIZE;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    float accsum = 0.0f;

    A = &A[(by * BM) * K];
    B = &B[(bx * BN)];
    C = &C[(by * BM) * N + bx * BN];

    for (int k = 0; k<K; k+= BK){
        As[ty * BK + tx] = A[ty * K + tx];
        Bs[ty * BN + tx] = B[ty * N + tx];

        __syncthreads();

        A = A + BK;
        B = B + BK * N;

        for (int i=0; i<BK; i++){
            accsum += As[ty * BK + i] * Bs[i * BN + tx];
        }

        __syncthreads();
    }

    output[ty*N + tx] = accsum;
    
 }

 const int BLOCK_SIZE = 32;
 dim3 block(BLOCK_SIZE, BLOCK_SIZE);
 dim3 grid(CEIL(N,BLOCK_SIZE), CEIL(M,BLOCK_SIZE));  // 根据C矩阵的形状(M行N列)切块
 sgemm<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

 // thread tile + block tile
 __global__ void sgemm(float* A, float* B, float* output, int M, int N, int K){

 }

 // softmax matrix 
__global__ void softmax(float* input, float* output, int M, int N){
    int lane_id = threadIdx.x % warp_size;

    __shared__ float s_max_val;
    __shared__ float s_sum;
    
    int row = blockIdx.x;
    if ( row >=M) return;

    int iteration = CEIL(N, warp_size);

    float max_val = -FLT_MAX;
    // 求得每一行的最大值
    for (int i=0; i < iteration; i++){
        int col = i*warp_size + lane_id;
        max_val = (col < N) ? fmaxf(max_val, input[row * N + col]) : max_val;
    }
    for (int offset = warp_size >> 1; offset > 0; offset>>=1){
        max_val = fmax(max_val, __shfl_down_synx(0xFFFFFFFF, max_val, offset))
    }
    if (lane_id == 0) s_max_val = max_val;

    //求得每一行的指数和的总和
    float sum = 0.0f;
    for (int i = 0; i < iteration; i++){
        int col = i * warp_size + lane_id;
        sum += (col < N) ? expf(inputp[row * N + col] - s_max_val) : 0.0f;
    }
    for (int offset = warp_size >> 1; offset > 0; offset >>= 1){
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    if (lane_id == 0) s_sum = sum;

    //计算每一行的softmax值
    for (int i= 0; i < iteration; i++){
        int col = i * warp_size + lane_id;
        if (col < N) output[row * N + col] = expf(inputp[row * N + col] - s_max_val) / s_sum;
    }


}

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <float.h>
#include <math.h>

static inline __host__ __device__ int ceil_div_int(int a, int b) { return (a + b - 1) / b; }

static inline __device__ float warp_allreduce_sum(float x) {
    for (int offset = 16; offset > 0; offset >>= 1) x += __shfl_down_sync(0xFFFFFFFFu, x, offset);
    return __shfl_sync(0xFFFFFFFFu, x, 0);
}

static inline __device__ float warp_allreduce_max(float x) {
    for (int offset = 16; offset > 0; offset >>= 1) x = fmaxf(x, __shfl_down_sync(0xFFFFFFFFu, x, offset));
    return __shfl_sync(0xFFFFFFFFu, x, 0);
}

template <int HEAD_DIM, int BLOCK_M, int BLOCK_N, bool CAUSAL>
__global__ void flash_attn_fwd_f16(const half* __restrict__ q,
                                  const half* __restrict__ k,
                                  const half* __restrict__ v,
                                  half* __restrict__ o,
                                  int seqlen_q,
                                  int seqlen_k) {
    constexpr int kWarpSize = 32;
    constexpr int kVec = 4;
    const int lane = threadIdx.x;
    const int q_row_in_block = threadIdx.y;
    const int q_row = blockIdx.x * BLOCK_M + q_row_in_block;
    const int bh = blockIdx.y;

    if (q_row >= seqlen_q) return;

    const half* q_ptr = q + (static_cast<int64_t>(bh) * seqlen_q + q_row) * HEAD_DIM;
    const half* k_ptr = k + static_cast<int64_t>(bh) * seqlen_k * HEAD_DIM;
    const half* v_ptr = v + static_cast<int64_t>(bh) * seqlen_k * HEAD_DIM;
    half* o_ptr = o + (static_cast<int64_t>(bh) * seqlen_q + q_row) * HEAD_DIM;

    __shared__ half smem_k[BLOCK_N * HEAD_DIM];
    __shared__ half smem_v[BLOCK_N * HEAD_DIM];

    float o_vec[kVec];
    #pragma unroll
    for (int i = 0; i < kVec; ++i) o_vec[i] = 0.0f;
    float m = -INFINITY;
    float l = 0.0f;
    const float scale = rsqrtf(static_cast<float>(HEAD_DIM));

    const int lane_offset = lane * kVec;
    float q_reg[kVec];
    #pragma unroll
    for (int i = 0; i < kVec; ++i) {
        const int d = lane_offset + i;
        q_reg[i] = (d < HEAD_DIM) ? __half2float(q_ptr[d]) : 0.0f;
    }

    for (int k0 = 0; k0 < seqlen_k; k0 += BLOCK_N) {
        const int threads = blockDim.x * blockDim.y;
        const int tid = threadIdx.y * blockDim.x + threadIdx.x;
        const int elements = BLOCK_N * HEAD_DIM;
        for (int idx = tid; idx < elements; idx += threads) {
            const int kk = idx / HEAD_DIM;
            const int d = idx - kk * HEAD_DIM;
            const int k_row = k0 + kk;
            half kval = __float2half(0.0f);
            half vval = __float2half(0.0f);
            if (k_row < seqlen_k) {
                kval = k_ptr[static_cast<int64_t>(k_row) * HEAD_DIM + d];
                vval = v_ptr[static_cast<int64_t>(k_row) * HEAD_DIM + d];
            }
            smem_k[idx] = kval;
            smem_v[idx] = vval;
        }
        __syncthreads();

        float m_block = -INFINITY;
        for (int j = 0; j < BLOCK_N; ++j) {
            const int k_col = k0 + j;
            if (k_col >= seqlen_k) break;
            float partial = 0.0f;
            #pragma unroll
            for (int i = 0; i < kVec; ++i) {
                const int d = lane_offset + i;
                if (d < HEAD_DIM) partial += q_reg[i] * __half2float(smem_k[j * HEAD_DIM + d]);
            }
            float dot = warp_allreduce_sum(partial);
            float score = dot * scale;
            if constexpr (CAUSAL) {
                if (k_col > q_row) score = -INFINITY;
            }
            m_block = fmaxf(m_block, score);
        }
        m_block = warp_allreduce_max(m_block);

        const float m_new = fmaxf(m, m_block);
        const float scale_old = expf(m - m_new);
        l *= scale_old;
        #pragma unroll
        for (int i = 0; i < kVec; ++i) o_vec[i] *= scale_old;

        float l_new = 0.0f;
        for (int j = 0; j < BLOCK_N; ++j) {
            const int k_col = k0 + j;
            if (k_col >= seqlen_k) break;
            float partial = 0.0f;
            #pragma unroll
            for (int i = 0; i < kVec; ++i) {
                const int d = lane_offset + i;
                if (d < HEAD_DIM) partial += q_reg[i] * __half2float(smem_k[j * HEAD_DIM + d]);
            }
            float dot = warp_allreduce_sum(partial);
            float score = dot * scale;
            if constexpr (CAUSAL) {
                if (k_col > q_row) score = -INFINITY;
            }
            const float p = expf(score - m_new);
            l_new += p;
            #pragma unroll
            for (int i = 0; i < kVec; ++i) {
                const int d = lane_offset + i;
                if (d < HEAD_DIM) o_vec[i] += p * __half2float(smem_v[j * HEAD_DIM + d]);
            }
        }
        l_new = __shfl_sync(0xFFFFFFFFu, l_new, 0);
        l += l_new;
        m = m_new;

        __syncthreads();
    }

    const float inv_l = 1.0f / l;
    #pragma unroll
    for (int i = 0; i < kVec; ++i) {
        const int d = lane_offset + i;
        if (d < HEAD_DIM) o_ptr[d] = __float2half_rn(o_vec[i] * inv_l);
    }
}

template <int BLOCK_SIZE>
__global__ void topk_bitonic_rowwise(const float* __restrict__ x,
                                    float* __restrict__ out_vals,
                                    int32_t* __restrict__ out_idx,
                                    int rows,
                                    int cols,
                                    int k) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int tid = threadIdx.x;

    __shared__ float s_val[BLOCK_SIZE];
    __shared__ int32_t s_idx[BLOCK_SIZE];

    float v = -INFINITY;
    int32_t i = -1;
    if (tid < cols) {
        v = x[static_cast<int64_t>(row) * cols + tid];
        i = tid;
    }
    s_val[tid] = v;
    s_idx[tid] = i;
    __syncthreads();

    for (unsigned int size = 2; size <= BLOCK_SIZE; size <<= 1) {
        for (unsigned int stride = size >> 1; stride > 0; stride >>= 1) {
            unsigned int pos = tid;
            unsigned int other = pos ^ stride;
            if (other > pos) {
                bool descending = ((pos & size) == 0);
                float a = s_val[pos], b = s_val[other];
                int32_t ia = s_idx[pos], ib = s_idx[other];
                bool swap = descending ? (a < b) : (a > b);
                if (swap) {
                    s_val[pos] = b; s_val[other] = a;
                    s_idx[pos] = ib; s_idx[other] = ia;
                }
            }
            __syncthreads();
        }
    }

    if (tid < k) {
        out_vals[static_cast<int64_t>(row) * k + tid] = s_val[tid];
        out_idx[static_cast<int64_t>(row) * k + tid] = s_idx[tid];
    }
}

template <int BLOCK_SIZE>
__global__ void scan_inclusive_block_i32(const int32_t* __restrict__ in,
                                        int32_t* __restrict__ out,
                                        int32_t* __restrict__ block_sums,
                                        int n) {
    __shared__ int32_t s_data[BLOCK_SIZE];
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * BLOCK_SIZE + tid;
    int32_t x = (gid < n) ? in[gid] : 0;
    s_data[tid] = x;
    __syncthreads();

    for (int offset = 1; offset < BLOCK_SIZE; offset <<= 1) {
        int32_t val = 0;
        if (tid >= offset) val = s_data[tid - offset];
        __syncthreads();
        s_data[tid] += val;
        __syncthreads();
    }

    if (gid < n) out[gid] = s_data[tid];
    if (block_sums && tid == BLOCK_SIZE - 1) block_sums[blockIdx.x] = s_data[tid];
}

__global__ void add_scan_block_offsets_i32(int32_t* __restrict__ out,
                                          const int32_t* __restrict__ scanned_block_sums,
                                          int n) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    const int b = blockIdx.x;
    const int32_t offset = (b == 0) ? 0 : scanned_block_sums[b - 1];
    out[gid] += offset;
}