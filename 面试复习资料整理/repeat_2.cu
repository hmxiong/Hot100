__global__ void reduce_v3(float* input, float* output, int M){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int warp_size = 32;

    int warp_id = threadIdx.x / warp_size;
    int lane_id = threadIdx.x % warp_size;

    __shared__ float smem[BLOCK_SIZE];

    float val = (idx < N) ? input[idx] : 0.0f;

    for (int offset = warp_size >> 1; offset>0; offset>>=1){
        val += __shfl_down_sync(0xFFFFFFFF, val , offset);
    }
    if (lane_id==0) smem[warp_id]=val;

    if(warp_id ==0){
        int warp_num = blockDim.x / warp_size;
        val = (lane_id < warp_num) ? smem[lane_id] : 0.0f;
        for (int offset = warp_num >> 1; offset >0; offset>>=1){
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        if (lane_id==0) atomicAdd(output, val);
    }
}

__global__ void softmax_matrix(float* input, float* output, int M , int N){
    int warp_size =32;
    int lane_id = threadIdx.x % warp_size;

    __shared__ float s_max_val;
    __shared__ float s_sum;

    int row = blockIdx.x;
    if (row < M) return;

    int iteration = CEIL(N, warp_size);

    float max_val = -FLT_MAX;
    for (i = 0; i< iteration, i++){
        col = i+warp_size + lane_id;
        max_val = (col < N)? fmaxf(max_val, input[row*N + col]) : max_val;
    }
    for (int offset = warp_size >> 1; offset > 0; offset >>= 1){
        max_val = fmaxf(max_val, __shfl_down_sync(0xFFFFFFFF, max_val, doffset));
    }
    if (lane_id ==0) s_max_val = max_val

    float sum = 0.0f;
    for (int i = 0; i < iteration; i++){
        col = i*warp_size + lane_id;
        sum  += __shfl_down_sync(0xFFFFFFFF, expf(input[row*N + col] - s_max_val))
    }
    for (int offset = warp_size>>1; offset>0; offset>>=1){
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    if(lane_id ==0) s_sum = sum;

    for (int i = 0; i< iteration; i++){
        int col = i*warp_size + lane_id;
        if (col < N) output[row*N + col] = expf(input[row*N + col] - s_max_val) / s_sum;
    }

}

__global__ void gemm_block_tile(float* A, float* B, float* C, int M, int N, int K){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    const int BM = BLOCK_SIZE;
    const int BK = BLOCK_SIZE;
    const int BN = BLOCK_SIZE;

    __shared__ float s_A[BM * BK];
    __shared__ float s_B[BK * BN];

    float acc = 0.0f;

    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BN * N + bx * BN];

    for (int k=0; k< K, k+=BK){
        s_A[tx * BK + ty] = A[tx * K + ty];
        s_B[ty * BN + tx] = B[ty * N + tx];
        __syncthreads();

        A = A + BK;
        B = B + BK * N;

        for (int i=0; i< BK;i++){
            acc += s_A[ty * BK + i] * s_B[i * BN + tx];
        }

        __syncthreads();
    }

    C[ty * BN + tx] = acc;

    
}