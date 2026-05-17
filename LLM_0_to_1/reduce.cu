dim3 block_size(BLOCK_SIZE);
dim3 grid_size(CEIL(N, BLOCK_SIZE));

__global__ void recude_v3(float* input, float* output, int N){
    constexpr int WARP_SIZE = 32;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int lane_id = threadIdx.x & (WARP_SIZE - 1);
    int warp_id = threadIdx.x >> 5;

    constexpr int SMEM_WARPS = (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float smem[SMEM_WARPS];

    float val = (idx < N) ? input[idx] : 0.0f;

    unsigned mask = __activemask();
    int active = __popc(mask);
    #pragma unroll
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(mask, val, offset, WARP_SIZE);
        if (lane_id + offset < active) val += other;
    }

    if (lane_id == 0) {
        smem[warp_id] = val;
    }

    __syncthreads();

    if (warp_id == 0) {
        int warp_num = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
        float sum = 0.0f;

        for (int base = 0; base < warp_num; base += WARP_SIZE) {
            int rem = warp_num - base;
            int take = rem < WARP_SIZE ? rem : WARP_SIZE;
            unsigned mask2 = take == WARP_SIZE ? 0xFFFFFFFFu : ((1u << take) - 1u);

            float v = (lane_id < take) ? smem[base + lane_id] : 0.0f;
            #pragma unroll
            for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
                float other = __shfl_down_sync(mask2, v, offset, WARP_SIZE);
                if (lane_id + offset < take) v += other;
            }

            if (lane_id == 0) sum += v;
        }

        if (lane_id == 0) atomicAdd(output, sum);
    }
}

recude_v3<<<grid_size, block_size>>>(input, output, N);
