__global__ void reduce_v3(float* input, float* output, int N){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    int warp_id = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;

    __shared__ float s_mem[warpSize];

    float val = (idx < N) ? input[idx] : 0.0f;

    for(int offset = warpSize>>1; offset > 0; offset>>=1){
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    if(lane_id == 0){
        s_mem[warp_id] = val;
    }

    __syncthreads();

    if (warp_id == 0){
        int wapr_num = blockDim.x / warpSize;
        val = (lane_id < wapr_num) ? s_mem[lane_id] : 0.0f;
        for(int offset = warpSize>>1; offset>1; offset>>=1){
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        if(lane_id==0) atomicAdd(output, val);
    }
}

