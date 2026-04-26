Skip to content
 
Search Gists
Search...
All gists
Back to GitHub
@lexpank
lexpank/microgpt.cu
Created last month • Report abuse
Code
Revisions
1
Stars
3
Clone this repository at &lt;script src=&quot;https://gist.github.com/lexpank/b4074152238b21a49948acf0417c9e6f.js&quot;&gt;&lt;/script&gt;
<script src="https://gist.github.com/lexpank/b4074152238b21a49948acf0417c9e6f.js"></script>
CUDA C microgpt
microgpt.cu
// Nobody asked for microgpt in CUDA C, which is exactly why it had to happen.
// It is no longer particularly micro.
// It is, however, stupidly fast.

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_LAYER 1
#define N_EMBD 16
#define BLOCK_SIZE 16
#define N_HEAD 4
#define HEAD_DIM (N_EMBD / N_HEAD)
#define MLP_HIDDEN (4 * N_EMBD)
#define INIT_STD 0.08f
#define RMS_EPS 1e-5f
#define LEARNING_RATE 0.01f
#define BETA1 0.85f
#define BETA2 0.99f
#define ADAM_EPS 1e-8f

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err__));                                 \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

#define CHECK_CUBLAS(call)                                                      \
    do {                                                                        \
        cublasStatus_t status__ = (call);                                       \
        if (status__ != CUBLAS_STATUS_SUCCESS) {                                \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__,  \
                    (int)status__);                                             \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

typedef struct {
    char **items;
    int count;
    int capacity;
} StringArray;

typedef struct {
    StringArray docs;
    int vocab_size;
    int bos_id;
    int char_to_id[256];
    unsigned char id_to_char[256];
} Dataset;

typedef struct {
    const char *name;
    int rows;
    int cols;
    float *data;
    float *grad;
    float *m;
    float *v;
} Param;

typedef struct {
    float *x_in;
    float *attn_norm;
    float *q;
    float *k;
    float *v;
    float *attn_logits;
    float *attn_weights;
    float *attn_out;
    float *attn_proj;
    float *mlp_in;
    float *mlp_norm;
    float *fc1;
    float *relu;
    float *fc2;
    float *x_out;
} LayerCache;

typedef struct {
    int token_id;
    int target_id;
    float *embed_in;
    float *x0;
    float *logits;
    float *probs;
    LayerCache layers[N_LAYER];
} PositionCache;

typedef struct {
    PositionCache pos[BLOCK_SIZE];
    float *keys[N_LAYER];
    float *values[N_LAYER];
    float *grad_keys[N_LAYER];
    float *grad_values[N_LAYER];
    float *grad_k_accum[N_LAYER][BLOCK_SIZE];
    float *grad_v_accum[N_LAYER][BLOCK_SIZE];
} TrainCache;

typedef struct {
    float *grad_logits;
    float *grad_x;
    float *grad_x_in;
    float *grad_mlp_in;
    float *grad_relu;
    float *grad_fc1;
    float *grad_mlp_norm;
    float *grad_norm_tmp;
    float *grad_attn_out;
    float *grad_q;
    float *grad_attn_norm;
    float *grad_embed_in;
} Scratch;

typedef struct {
    Param wte;
    Param wpe;
    Param lm_head;
    Param attn_wq[N_LAYER];
    Param attn_wk[N_LAYER];
    Param attn_wv[N_LAYER];
    Param attn_wo[N_LAYER];
    Param mlp_fc1[N_LAYER];
    Param mlp_fc2[N_LAYER];
} Model;

typedef struct {
    int num_steps;
    int num_samples;
    float temperature;
    const char *input_path;
} Options;

typedef struct {
    float *base;
    size_t count;
    size_t offset;
} FloatArena;

static cublasHandle_t g_cublas = NULL;

static __global__ void vec_add_kernel(const float *a, const float *b, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

static __global__ void vec_add_inplace_kernel(float *dst, const float *src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] += src[idx];
    }
}

static __global__ void relu_forward_kernel(const float *x, float *y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = x[idx] > 0.0f ? x[idx] : 0.0f;
    }
}

static __global__ void relu_backward_kernel(const float *x, const float *grad_y, float *grad_x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad_x[idx] = x[idx] > 0.0f ? grad_y[idx] : 0.0f;
    }
}

static __global__ void rmsnorm_forward_kernel(const float *x, float *y, int n) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float mean_square = 0.0f;
        for (int i = 0; i < n; ++i) {
            mean_square += x[i] * x[i];
        }
        mean_square /= (float)n;
        float scale = rsqrtf(mean_square + RMS_EPS);
        for (int i = 0; i < n; ++i) {
            y[i] = x[i] * scale;
        }
    }
}

static __global__ void rmsnorm_backward_kernel(const float *x, const float *grad_y, float *grad_x, int n) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float mean_square = 0.0f;
        float dot = 0.0f;
        for (int i = 0; i < n; ++i) {
            mean_square += x[i] * x[i];
            dot += grad_y[i] * x[i];
        }
        mean_square /= (float)n;
        float scale = rsqrtf(mean_square + RMS_EPS);
        float scale_cubed = scale * scale * scale;
        for (int i = 0; i < n; ++i) {
            grad_x[i] = grad_y[i] * scale - x[i] * dot * scale_cubed / (float)n;
        }
    }
}

static __global__ void scaled_softmax_kernel(const float *logits, float *probs, float scale, int n) {
    __shared__ float scratch[256];
    int tid = threadIdx.x;

    float local_max = -1e30f;
    for (int i = tid; i < n; i += blockDim.x) {
        float v = logits[i] * scale;
        if (v > local_max) {
            local_max = v;
        }
    }
    scratch[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && scratch[tid + stride] > scratch[tid]) {
            scratch[tid] = scratch[tid + stride];
        }
        __syncthreads();
    }

    float max_val = scratch[0];
    float local_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float e = expf(logits[i] * scale - max_val);
        probs[i] = e;
        local_sum += e;
    }
    scratch[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }

    float inv_sum = 1.0f / scratch[0];
    for (int i = tid; i < n; i += blockDim.x) {
        probs[i] *= inv_sum;
    }
}

static __global__ void cross_entropy_backward_kernel(const float *probs, float *grad_logits, int target_id, float loss_scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float grad = probs[idx] * loss_scale;
        if (idx == target_id) {
            grad -= loss_scale;
        }
        grad_logits[idx] = grad;
    }
}

static __global__ void adam_update_kernel(
    float *data,
    float *grad,
    float *m,
    float *v,
    float lr_t,
    float beta1_corr,
    float beta2_corr,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = grad[idx];
        float m_new = BETA1 * m[idx] + (1.0f - BETA1) * g;
        float v_new = BETA2 * v[idx] + (1.0f - BETA2) * g * g;
        m[idx] = m_new;
        v[idx] = v_new;
        float m_hat = m_new / beta1_corr;
        float v_hat = v_new / beta2_corr;
        data[idx] -= lr_t * m_hat / (sqrtf(v_hat) + ADAM_EPS);
        grad[idx] = 0.0f;
    }
}

static __global__ void row_add_kernel(float *row_grad, const float *grad, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        row_grad[idx] += grad[idx];
    }
}

static __global__ void attention_forward_kernel(
    const float *q,
    const float *keys,
    const float *values,
    float *attn_logits,
    float *attn_weights,
    float *attn_out,
    int pos_id
) {
    int h = threadIdx.x;
    if (h >= N_HEAD) {
        return;
    }

    int hs = h * HEAD_DIM;
    float logits_local[BLOCK_SIZE];
    float weights_local[BLOCK_SIZE];
    float max_val = -1e30f;
    float sum = 0.0f;
    float scale = 1.0f / sqrtf((float)HEAD_DIM);

    for (int t = 0; t <= pos_id; ++t) {
        float dot = 0.0f;
        const float *k_t = keys + (size_t)t * N_EMBD + hs;
        for (int j = 0; j < HEAD_DIM; ++j) {
            dot += q[hs + j] * k_t[j];
        }
        logits_local[t] = dot * scale;
        if (logits_local[t] > max_val) {
            max_val = logits_local[t];
        }
    }

    for (int t = 0; t <= pos_id; ++t) {
        float w = expf(logits_local[t] - max_val);
        weights_local[t] = w;
        sum += w;
    }

    float inv_sum = 1.0f / sum;
    for (int t = 0; t <= pos_id; ++t) {
        float w = weights_local[t] * inv_sum;
        attn_logits[h * BLOCK_SIZE + t] = logits_local[t];
        attn_weights[h * BLOCK_SIZE + t] = w;
        weights_local[t] = w;
    }

    for (int j = 0; j < HEAD_DIM; ++j) {
        float out = 0.0f;
        for (int t = 0; t <= pos_id; ++t) {
            const float *v_t = values + (size_t)t * N_EMBD + hs;
            out += weights_local[t] * v_t[j];
        }
        attn_out[hs + j] = out;
    }
}

static __global__ void attention_backward_kernel(
    const float *q,
    const float *keys,
    const float *values,
    const float *attn_weights,
    const float *grad_attn_out,
    float *grad_q,
    float *grad_keys,
    float *grad_values,
    int pos_id
) {
    int h = threadIdx.x;
    if (h >= N_HEAD) {
        return;
    }

    int hs = h * HEAD_DIM;
    float grad_w[BLOCK_SIZE];
    float grad_q_local[HEAD_DIM];
    float weighted_sum = 0.0f;
    float scale = 1.0f / sqrtf((float)HEAD_DIM);

    for (int j = 0; j < HEAD_DIM; ++j) {
        grad_q_local[j] = 0.0f;
    }

    for (int t = 0; t <= pos_id; ++t) {
        const float *v_t = values + (size_t)t * N_EMBD + hs;
        float dot = 0.0f;
        float weight = attn_weights[h * BLOCK_SIZE + t];
        float *grad_v_t = grad_values + (size_t)t * N_EMBD + hs;
        for (int j = 0; j < HEAD_DIM; ++j) {
            float go = grad_attn_out[hs + j];
            dot += go * v_t[j];
            grad_v_t[j] += go * weight;
        }
        grad_w[t] = dot;
        weighted_sum += dot * weight;
    }

    for (int t = 0; t <= pos_id; ++t) {
        const float *k_t = keys + (size_t)t * N_EMBD + hs;
        float *grad_k_t = grad_keys + (size_t)t * N_EMBD + hs;
        float weight = attn_weights[h * BLOCK_SIZE + t];
        float grad_logit = weight * (grad_w[t] - weighted_sum);
        for (int j = 0; j < HEAD_DIM; ++j) {
            grad_q_local[j] += grad_logit * k_t[j] * scale;
            grad_k_t[j] += grad_logit * q[hs + j] * scale;
        }
    }

    for (int j = 0; j < HEAD_DIM; ++j) {
        grad_q[hs + j] = grad_q_local[j];
    }
}

static void zero_device(float *x, int n);
static void copy_device(float *dst, const float *src, int n);
static void copy_to_host(float *dst, const float *src, int n);
static void copy_to_device(float *dst, const float *src, int n);

static void launch_vec_add(const float *a, const float *b, float *out, int n) {
    int threads = n < 256 ? n : 256;
    vec_add_kernel<<<1, threads>>>(a, b, out, n);
    CHECK_CUDA(cudaGetLastError());
}

static void launch_vec_add_inplace(float *dst, const float *src, int n) {
    int threads = n < 256 ? n : 256;
    vec_add_inplace_kernel<<<1, threads>>>(dst, src, n);
    CHECK_CUDA(cudaGetLastError());
}

static void launch_relu_forward(const float *x, float *y, int n) {
    int threads = n < 256 ? n : 256;
    relu_forward_kernel<<<1, threads>>>(x, y, n);
    CHECK_CUDA(cudaGetLastError());
}

static void launch_relu_backward(const float *x, const float *grad_y, float *grad_x, int n) {
    int threads = n < 256 ? n : 256;
    relu_backward_kernel<<<1, threads>>>(x, grad_y, grad_x, n);
    CHECK_CUDA(cudaGetLastError());
}

static void launch_matvec(const float *w, const float *x, float *y, int rows, int cols) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CHECK_CUBLAS(cublasSgemv(
        g_cublas,
        CUBLAS_OP_T,
        cols,
        rows,
        &alpha,
        w,
        cols,
        x,
        1,
        &beta,
        y,
        1
    ));
}

static void launch_linear_backward(
    const float *w,
    const float *x,
    const float *grad_out,
    float *grad_w,
    float *grad_x,
    int rows,
    int cols
) {
    const float alpha = 1.0f;
    const float beta = 1.0f;

    CHECK_CUBLAS(cublasSger(
        g_cublas,
        cols,
        rows,
        &alpha,
        x,
        1,
        grad_out,
        1,
        grad_w,
        cols
    ));

    CHECK_CUBLAS(cublasSgemv(
        g_cublas,
        CUBLAS_OP_N,
        cols,
        rows,
        &alpha,
        w,
        cols,
        grad_out,
        1,
        &beta,
        grad_x,
        1
    ));
}

static void launch_rmsnorm_forward(const float *x, float *y, int n) {
    rmsnorm_forward_kernel<<<1, 1>>>(x, y, n);
    CHECK_CUDA(cudaGetLastError());
}

static void launch_rmsnorm_backward(const float *x, const float *grad_y, float *grad_x, int n) {
    rmsnorm_backward_kernel<<<1, 1>>>(x, grad_y, grad_x, n);
    CHECK_CUDA(cudaGetLastError());
}

static void launch_scaled_softmax(const float *logits, float *probs, float scale, int n) {
    scaled_softmax_kernel<<<1, 256>>>(logits, probs, scale, n);
    CHECK_CUDA(cudaGetLastError());
}

static void launch_cross_entropy_backward(const float *probs, float *grad_logits, int target_id, float loss_scale, int n) {
    int threads = n < 256 ? n : 256;
    cross_entropy_backward_kernel<<<1, threads>>>(probs, grad_logits, target_id, loss_scale, n);
    CHECK_CUDA(cudaGetLastError());
}

static void launch_attention_forward(
    const float *q,
    const float *keys,
    const float *values,
    float *attn_logits,
    float *attn_weights,
    float *attn_out,
    int pos_id
) {
    attention_forward_kernel<<<1, N_HEAD>>>(q, keys, values, attn_logits, attn_weights, attn_out, pos_id);
    CHECK_CUDA(cudaGetLastError());
}

static void launch_attention_backward(
    const float *q,
    const float *keys,
    const float *values,
    const float *attn_weights,
    const float *grad_attn_out,
    float *grad_q,
    float *grad_keys,
    float *grad_values,
    int pos_id
) {
    attention_backward_kernel<<<1, N_HEAD>>>(
        q,
        keys,
        values,
        attn_weights,
        grad_attn_out,
        grad_q,
        grad_keys,
        grad_values,
        pos_id
    );
    CHECK_CUDA(cudaGetLastError());
}

static void launch_adam_update(float *data, float *grad, float *m, float *v, float lr_t, float beta1_corr, float beta2_corr, int n) {
    int threads = n < 256 ? n : 256;
    int blocks = (n + threads - 1) / threads;
    adam_update_kernel<<<blocks, threads>>>(data, grad, m, v, lr_t, beta1_corr, beta2_corr, n);
    CHECK_CUDA(cudaGetLastError());
}

static void launch_row_add(float *row_grad, const float *grad, int n) {
    int threads = n < 256 ? n : 256;
    row_add_kernel<<<1, threads>>>(row_grad, grad, n);
    CHECK_CUDA(cudaGetLastError());
}

static void *xmalloc(size_t size) {
    void *ptr = malloc(size);
    if (ptr == NULL) {
        fprintf(stderr, "malloc failed\n");
        exit(1);
    }
    return ptr;
}

static char *xstrdup(const char *src) {
    size_t len = strlen(src);
    char *dst = (char *)xmalloc(len + 1);
    memcpy(dst, src, len + 1);
    return dst;
}

static void init_arena(FloatArena *arena, size_t count) {
    arena->count = count;
    arena->offset = 0;
    CHECK_CUDA(cudaMalloc((void **)&arena->base, count * sizeof(float)));
    CHECK_CUDA(cudaMemset(arena->base, 0, count * sizeof(float)));
}

static float *arena_alloc(FloatArena *arena, size_t count) {
    if (arena->offset + count > arena->count) {
        fprintf(stderr, "arena overflow: requested %zu floats, %zu remaining\n",
                count, arena->count - arena->offset);
        exit(1);
    }
    float *ptr = arena->base + arena->offset;
    arena->offset += count;
    CHECK_CUDA(cudaMemset(ptr, 0, count * sizeof(float)));
    return ptr;
}

static void init_string_array(StringArray *arr) {
    arr->items = NULL;
    arr->count = 0;
    arr->capacity = 0;
}

static void push_string(StringArray *arr, const char *value) {
    if (arr->count == arr->capacity) {
        int new_capacity = arr->capacity == 0 ? 64 : arr->capacity * 2;
        char **new_items = (char **)realloc(arr->items, (size_t)new_capacity * sizeof(char *));
        if (new_items == NULL) {
            fprintf(stderr, "realloc failed\n");
            exit(1);
        }
        arr->items = new_items;
        arr->capacity = new_capacity;
    }
    arr->items[arr->count++] = xstrdup(value);
}

static void shuffle_docs(StringArray *arr) {
    for (int i = arr->count - 1; i > 0; --i) {
        int j = rand() % (i + 1);
        char *tmp = arr->items[i];
        arr->items[i] = arr->items[j];
        arr->items[j] = tmp;
    }
}

static void load_dataset(Dataset *dataset, const char *input_path) {
    init_string_array(&dataset->docs);
    for (int i = 0; i < 256; ++i) {
        dataset->char_to_id[i] = -1;
        dataset->id_to_char[i] = 0;
    }

    FILE *fp = fopen(input_path, "r");
    if (fp == NULL) {
        fprintf(stderr, "could not open %s\n", input_path);
        exit(1);
    }

    char *line = NULL;
    size_t cap = 0;
    ssize_t read = 0;
    while ((read = getline(&line, &cap, fp)) != -1) {
        while (read > 0 && (line[read - 1] == '\n' || line[read - 1] == '\r')) {
            line[--read] = '\0';
        }
        if (read > 0) {
            push_string(&dataset->docs, line);
        }
    }
    free(line);
    fclose(fp);

    shuffle_docs(&dataset->docs);

    int vocab = 0;
    int seen[256] = {0};
    for (int d = 0; d < dataset->docs.count; ++d) {
        const unsigned char *doc = (const unsigned char *)dataset->docs.items[d];
        for (int i = 0; doc[i] != '\0'; ++i) {
            seen[doc[i]] = 1;
        }
    }
    for (int c = 0; c < 256; ++c) {
        if (seen[c]) {
            dataset->char_to_id[c] = vocab;
            dataset->id_to_char[vocab] = (unsigned char)c;
            ++vocab;
        }
    }

    dataset->bos_id = vocab;
    dataset->vocab_size = vocab + 1;
}

static float rand_uniform(void) {
    return ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
}

static float rand_normal(float stddev) {
    float u1 = rand_uniform();
    float u2 = rand_uniform();
    float mag = sqrtf(-2.0f * logf(u1));
    float phase = 2.0f * (float)M_PI * u2;
    return stddev * mag * cosf(phase);
}

static void init_param(Param *param, FloatArena *arena, const char *name, int rows, int cols) {
    size_t count = (size_t)rows * (size_t)cols;
    float *host_data = (float *)xmalloc(count * sizeof(float));
    param->name = name;
    param->rows = rows;
    param->cols = cols;
    param->data = arena_alloc(arena, count);
    param->grad = arena_alloc(arena, count);
    param->m = arena_alloc(arena, count);
    param->v = arena_alloc(arena, count);
    for (size_t i = 0; i < count; ++i) {
        host_data[i] = rand_normal(INIT_STD);
    }
    copy_to_device(param->data, host_data, (int)count);
    free(host_data);
}

static size_t param_count(const Param *param) {
    return (size_t)param->rows * (size_t)param->cols;
}

static size_t param_storage_count(int vocab_size) {
    size_t count = 0;
    count += 4 * (size_t)vocab_size * N_EMBD;
    count += 4 * (size_t)BLOCK_SIZE * N_EMBD;
    count += 4 * (size_t)vocab_size * N_EMBD;
    for (int layer = 0; layer < N_LAYER; ++layer) {
        count += 4 * (size_t)N_EMBD * N_EMBD;
        count += 4 * (size_t)N_EMBD * N_EMBD;
        count += 4 * (size_t)N_EMBD * N_EMBD;
        count += 4 * (size_t)N_EMBD * N_EMBD;
        count += 4 * (size_t)MLP_HIDDEN * N_EMBD;
        count += 4 * (size_t)N_EMBD * MLP_HIDDEN;
    }
    return count;
}

static void init_model(Model *model, FloatArena *arena, int vocab_size) {
    init_param(&model->wte, arena, "wte", vocab_size, N_EMBD);
    init_param(&model->wpe, arena, "wpe", BLOCK_SIZE, N_EMBD);
    init_param(&model->lm_head, arena, "lm_head", vocab_size, N_EMBD);
    for (int layer = 0; layer < N_LAYER; ++layer) {
        init_param(&model->attn_wq[layer], arena, "attn_wq", N_EMBD, N_EMBD);
        init_param(&model->attn_wk[layer], arena, "attn_wk", N_EMBD, N_EMBD);
        init_param(&model->attn_wv[layer], arena, "attn_wv", N_EMBD, N_EMBD);
        init_param(&model->attn_wo[layer], arena, "attn_wo", N_EMBD, N_EMBD);
        init_param(&model->mlp_fc1[layer], arena, "mlp_fc1", MLP_HIDDEN, N_EMBD);
        init_param(&model->mlp_fc2[layer], arena, "mlp_fc2", N_EMBD, MLP_HIDDEN);
    }
}

static size_t total_params(const Model *model) {
    size_t count = 0;
    count += param_count(&model->wte);
    count += param_count(&model->wpe);
    count += param_count(&model->lm_head);
    for (int layer = 0; layer < N_LAYER; ++layer) {
        count += param_count(&model->attn_wq[layer]);
        count += param_count(&model->attn_wk[layer]);
        count += param_count(&model->attn_wv[layer]);
        count += param_count(&model->attn_wo[layer]);
        count += param_count(&model->mlp_fc1[layer]);
        count += param_count(&model->mlp_fc2[layer]);
    }
    return count;
}

static size_t layer_cache_storage_count(void) {
    size_t count = 0;
    count += N_EMBD;
    count += N_EMBD;
    count += N_EMBD;
    count += (size_t)N_HEAD * BLOCK_SIZE;
    count += (size_t)N_HEAD * BLOCK_SIZE;
    count += N_EMBD;
    count += N_EMBD;
    count += N_EMBD;
    count += N_EMBD;
    count += MLP_HIDDEN;
    count += MLP_HIDDEN;
    count += N_EMBD;
    count += N_EMBD;
    return count;
}

static void alloc_layer_cache(LayerCache *cache, FloatArena *arena) {
    cache->x_in = arena_alloc(arena, N_EMBD);
    cache->attn_norm = arena_alloc(arena, N_EMBD);
    cache->q = arena_alloc(arena, N_EMBD);
    cache->k = NULL;
    cache->v = NULL;
    cache->attn_logits = arena_alloc(arena, (size_t)N_HEAD * BLOCK_SIZE);
    cache->attn_weights = arena_alloc(arena, (size_t)N_HEAD * BLOCK_SIZE);
    cache->attn_out = arena_alloc(arena, N_EMBD);
    cache->attn_proj = arena_alloc(arena, N_EMBD);
    cache->mlp_in = arena_alloc(arena, N_EMBD);
    cache->mlp_norm = arena_alloc(arena, N_EMBD);
    cache->fc1 = arena_alloc(arena, MLP_HIDDEN);
    cache->relu = arena_alloc(arena, MLP_HIDDEN);
    cache->fc2 = arena_alloc(arena, N_EMBD);
    cache->x_out = arena_alloc(arena, N_EMBD);
}

static size_t train_cache_storage_count(int vocab_size) {
    size_t count = 0;
    for (int layer = 0; layer < N_LAYER; ++layer) {
        count += (size_t)BLOCK_SIZE * N_EMBD;
        count += (size_t)BLOCK_SIZE * N_EMBD;
        count += (size_t)BLOCK_SIZE * N_EMBD;
        count += (size_t)BLOCK_SIZE * N_EMBD;
    }
    for (int pos = 0; pos < BLOCK_SIZE; ++pos) {
        count += N_EMBD;
        count += N_EMBD;
        count += vocab_size;
        count += vocab_size;
        for (int layer = 0; layer < N_LAYER; ++layer) {
            count += layer_cache_storage_count();
        }
    }
    return count;
}

static void init_train_cache(TrainCache *cache, int vocab_size, FloatArena *arena) {
    for (int layer = 0; layer < N_LAYER; ++layer) {
        cache->keys[layer] = arena_alloc(arena, (size_t)BLOCK_SIZE * N_EMBD);
        cache->values[layer] = arena_alloc(arena, (size_t)BLOCK_SIZE * N_EMBD);
        cache->grad_keys[layer] = arena_alloc(arena, (size_t)BLOCK_SIZE * N_EMBD);
        cache->grad_values[layer] = arena_alloc(arena, (size_t)BLOCK_SIZE * N_EMBD);
    }

    for (int pos = 0; pos < BLOCK_SIZE; ++pos) {
        cache->pos[pos].embed_in = arena_alloc(arena, N_EMBD);
        cache->pos[pos].x0 = arena_alloc(arena, N_EMBD);
        cache->pos[pos].logits = arena_alloc(arena, vocab_size);
        cache->pos[pos].probs = arena_alloc(arena, vocab_size);
        for (int layer = 0; layer < N_LAYER; ++layer) {
            alloc_layer_cache(&cache->pos[pos].layers[layer], arena);
            cache->pos[pos].layers[layer].k = cache->keys[layer] + (size_t)pos * N_EMBD;
            cache->pos[pos].layers[layer].v = cache->values[layer] + (size_t)pos * N_EMBD;
            cache->grad_k_accum[layer][pos] = cache->grad_keys[layer] + (size_t)pos * N_EMBD;
            cache->grad_v_accum[layer][pos] = cache->grad_values[layer] + (size_t)pos * N_EMBD;
        }
    }
}

static size_t scratch_storage_count(int vocab_size) {
    size_t count = 0;
    count += vocab_size;
    count += N_EMBD;
    count += N_EMBD;
    count += N_EMBD;
    count += MLP_HIDDEN;
    count += MLP_HIDDEN;
    count += N_EMBD;
    count += N_EMBD;
    count += N_EMBD;
    count += N_EMBD;
    count += N_EMBD;
    count += N_EMBD;
    return count;
}

static void init_scratch(Scratch *scratch, int vocab_size, FloatArena *arena) {
    scratch->grad_logits = arena_alloc(arena, vocab_size);
    scratch->grad_x = arena_alloc(arena, N_EMBD);
    scratch->grad_x_in = arena_alloc(arena, N_EMBD);
    scratch->grad_mlp_in = arena_alloc(arena, N_EMBD);
    scratch->grad_relu = arena_alloc(arena, MLP_HIDDEN);
    scratch->grad_fc1 = arena_alloc(arena, MLP_HIDDEN);
    scratch->grad_mlp_norm = arena_alloc(arena, N_EMBD);
    scratch->grad_norm_tmp = arena_alloc(arena, N_EMBD);
    scratch->grad_attn_out = arena_alloc(arena, N_EMBD);
    scratch->grad_q = arena_alloc(arena, N_EMBD);
    scratch->grad_attn_norm = arena_alloc(arena, N_EMBD);
    scratch->grad_embed_in = arena_alloc(arena, N_EMBD);
}

static void zero_device(float *x, int n) {
    CHECK_CUDA(cudaMemset(x, 0, (size_t)n * sizeof(float)));
}

static void copy_device(float *dst, const float *src, int n) {
    CHECK_CUDA(cudaMemcpy(dst, src, (size_t)n * sizeof(float), cudaMemcpyDeviceToDevice));
}

static void copy_to_host(float *dst, const float *src, int n) {
    CHECK_CUDA(cudaMemcpy(dst, src, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost));
}

static void copy_to_device(float *dst, const float *src, int n) {
    CHECK_CUDA(cudaMemcpy(dst, src, (size_t)n * sizeof(float), cudaMemcpyHostToDevice));
}

static int tokenize_doc(const Dataset *dataset, const char *doc, int *tokens) {
    int count = 0;
    tokens[count++] = dataset->bos_id;
    for (int i = 0; doc[i] != '\0' && count < BLOCK_SIZE + 1; ++i) {
        tokens[count++] = dataset->char_to_id[(unsigned char)doc[i]];
    }
    if (count < BLOCK_SIZE + 2) {
        tokens[count++] = dataset->bos_id;
    }
    return count;
}

static const float *layer_input_ptr(const PositionCache *pos, int layer) {
    if (layer == 0) {
        return pos->x0;
    }
    return pos->layers[layer - 1].x_out;
}

static const float *final_hidden_ptr(const PositionCache *pos) {
    if (N_LAYER == 0) {
        return pos->x0;
    }
    return pos->layers[N_LAYER - 1].x_out;
}

static void forward_position(Model *model, TrainCache *cache, int vocab_size, int token_id, int pos_id) {
    PositionCache *pos = &cache->pos[pos_id];
    pos->token_id = token_id;

    const float *tok_emb = model->wte.data + (size_t)token_id * N_EMBD;
    const float *pos_emb = model->wpe.data + (size_t)pos_id * N_EMBD;
    launch_vec_add(tok_emb, pos_emb, pos->embed_in, N_EMBD);
    launch_rmsnorm_forward(pos->embed_in, pos->x0, N_EMBD);

    for (int layer = 0; layer < N_LAYER; ++layer) {
        LayerCache *lc = &pos->layers[layer];
        const float *x_in = layer_input_ptr(pos, layer);
        copy_device(lc->x_in, x_in, N_EMBD);

        launch_rmsnorm_forward(lc->x_in, lc->attn_norm, N_EMBD);
        launch_matvec(model->attn_wq[layer].data, lc->attn_norm, lc->q, N_EMBD, N_EMBD);
        launch_matvec(model->attn_wk[layer].data, lc->attn_norm, lc->k, N_EMBD, N_EMBD);
        launch_matvec(model->attn_wv[layer].data, lc->attn_norm, lc->v, N_EMBD, N_EMBD);
        launch_attention_forward(
            lc->q,
            cache->keys[layer],
            cache->values[layer],
            lc->attn_logits,
            lc->attn_weights,
            lc->attn_out,
            pos_id
        );

        launch_matvec(model->attn_wo[layer].data, lc->attn_out, lc->attn_proj, N_EMBD, N_EMBD);
        launch_vec_add(lc->attn_proj, lc->x_in, lc->mlp_in, N_EMBD);

        launch_rmsnorm_forward(lc->mlp_in, lc->mlp_norm, N_EMBD);
        launch_matvec(model->mlp_fc1[layer].data, lc->mlp_norm, lc->fc1, MLP_HIDDEN, N_EMBD);
        launch_relu_forward(lc->fc1, lc->relu, MLP_HIDDEN);
        launch_matvec(model->mlp_fc2[layer].data, lc->relu, lc->fc2, N_EMBD, MLP_HIDDEN);
        launch_vec_add(lc->fc2, lc->mlp_in, lc->x_out, N_EMBD);
    }

    launch_matvec(model->lm_head.data, final_hidden_ptr(pos), pos->logits, vocab_size, N_EMBD);
}

static float forward_sequence(Model *model, TrainCache *cache, const Dataset *dataset, const int *tokens, int n) {
    float total_loss = 0.0f;
    float *host_probs = (float *)xmalloc((size_t)dataset->vocab_size * sizeof(float));
    for (int pos = 0; pos < n; ++pos) {
        PositionCache *pc = &cache->pos[pos];
        pc->target_id = tokens[pos + 1];
        forward_position(model, cache, dataset->vocab_size, tokens[pos], pos);
        launch_scaled_softmax(pc->logits, pc->probs, 1.0f, dataset->vocab_size);
        copy_to_host(host_probs, pc->probs, dataset->vocab_size);
        float prob = host_probs[pc->target_id];
        if (prob < 1e-20f) {
            prob = 1e-20f;
        }
        total_loss += -logf(prob);
    }
    free(host_probs);
    return total_loss / (float)n;
}

static void zero_attention_accums(TrainCache *cache) {
    for (int layer = 0; layer < N_LAYER; ++layer) {
        CHECK_CUDA(cudaMemset(cache->grad_keys[layer], 0, (size_t)BLOCK_SIZE * N_EMBD * sizeof(float)));
        CHECK_CUDA(cudaMemset(cache->grad_values[layer], 0, (size_t)BLOCK_SIZE * N_EMBD * sizeof(float)));
    }
}

static void backward_sequence(Model *model, TrainCache *cache, Scratch *scratch, const Dataset *dataset, int n) {
    zero_attention_accums(cache);
    float loss_scale = 1.0f / (float)n;

    for (int pos = n - 1; pos >= 0; --pos) {
        PositionCache *pc = &cache->pos[pos];
        launch_cross_entropy_backward(pc->probs, scratch->grad_logits, pc->target_id, loss_scale, dataset->vocab_size);

        zero_device(scratch->grad_x, N_EMBD);
        launch_linear_backward(
            model->lm_head.data,
            final_hidden_ptr(pc),
            scratch->grad_logits,
            model->lm_head.grad,
            scratch->grad_x,
            dataset->vocab_size,
            N_EMBD
        );

        for (int layer = N_LAYER - 1; layer >= 0; --layer) {
            LayerCache *lc = &pc->layers[layer];

            copy_device(scratch->grad_mlp_in, scratch->grad_x, N_EMBD);
            zero_device(scratch->grad_relu, MLP_HIDDEN);
            launch_linear_backward(
                model->mlp_fc2[layer].data,
                lc->relu,
                scratch->grad_x,
                model->mlp_fc2[layer].grad,
                scratch->grad_relu,
                N_EMBD,
                MLP_HIDDEN
            );

            zero_device(scratch->grad_fc1, MLP_HIDDEN);
            launch_relu_backward(lc->fc1, scratch->grad_relu, scratch->grad_fc1, MLP_HIDDEN);

            zero_device(scratch->grad_mlp_norm, N_EMBD);
            launch_linear_backward(
                model->mlp_fc1[layer].data,
                lc->mlp_norm,
                scratch->grad_fc1,
                model->mlp_fc1[layer].grad,
                scratch->grad_mlp_norm,
                MLP_HIDDEN,
                N_EMBD
            );

            zero_device(scratch->grad_norm_tmp, N_EMBD);
            launch_rmsnorm_backward(lc->mlp_in, scratch->grad_mlp_norm, scratch->grad_norm_tmp, N_EMBD);
            launch_vec_add_inplace(scratch->grad_mlp_in, scratch->grad_norm_tmp, N_EMBD);

            copy_device(scratch->grad_x_in, scratch->grad_mlp_in, N_EMBD);
            zero_device(scratch->grad_attn_out, N_EMBD);
            launch_linear_backward(
                model->attn_wo[layer].data,
                lc->attn_out,
                scratch->grad_mlp_in,
                model->attn_wo[layer].grad,
                scratch->grad_attn_out,
                N_EMBD,
                N_EMBD
            );

            zero_device(scratch->grad_attn_norm, N_EMBD);
            launch_attention_backward(
                lc->q,
                cache->keys[layer],
                cache->values[layer],
                lc->attn_weights,
                scratch->grad_attn_out,
                scratch->grad_q,
                cache->grad_keys[layer],
                cache->grad_values[layer],
                pos
            );

            launch_linear_backward(
                model->attn_wq[layer].data,
                lc->attn_norm,
                scratch->grad_q,
                model->attn_wq[layer].grad,
                scratch->grad_attn_norm,
                N_EMBD,
                N_EMBD
            );
            launch_linear_backward(
                model->attn_wk[layer].data,
                lc->attn_norm,
                cache->grad_k_accum[layer][pos],
                model->attn_wk[layer].grad,
                scratch->grad_attn_norm,
                N_EMBD,
                N_EMBD
            );
            launch_linear_backward(
                model->attn_wv[layer].data,
                lc->attn_norm,
                cache->grad_v_accum[layer][pos],
                model->attn_wv[layer].grad,
                scratch->grad_attn_norm,
                N_EMBD,
                N_EMBD
            );

            zero_device(scratch->grad_norm_tmp, N_EMBD);
            launch_rmsnorm_backward(lc->x_in, scratch->grad_attn_norm, scratch->grad_norm_tmp, N_EMBD);
            launch_vec_add_inplace(scratch->grad_x_in, scratch->grad_norm_tmp, N_EMBD);
            copy_device(scratch->grad_x, scratch->grad_x_in, N_EMBD);
        }

        zero_device(scratch->grad_embed_in, N_EMBD);
        launch_rmsnorm_backward(pc->embed_in, scratch->grad_x, scratch->grad_embed_in, N_EMBD);
        launch_row_add(model->wte.grad + (size_t)pc->token_id * N_EMBD, scratch->grad_embed_in, N_EMBD);
        launch_row_add(model->wpe.grad + (size_t)pos * N_EMBD, scratch->grad_embed_in, N_EMBD);
    }
}

static void adam_update_param(Param *param, int step, int total_steps, float beta1_pow, float beta2_pow) {
    float lr_t = LEARNING_RATE * (1.0f - (float)step / (float)total_steps);
    int count = (int)param_count(param);
    launch_adam_update(param->data, param->grad, param->m, param->v, lr_t, 1.0f - beta1_pow, 1.0f - beta2_pow, count);
}

static void adam_update_model(Model *model, int step, int total_steps) {
    float beta1_pow = powf(BETA1, (float)(step + 1));
    float beta2_pow = powf(BETA2, (float)(step + 1));

    adam_update_param(&model->wte, step, total_steps, beta1_pow, beta2_pow);
    adam_update_param(&model->wpe, step, total_steps, beta1_pow, beta2_pow);
    adam_update_param(&model->lm_head, step, total_steps, beta1_pow, beta2_pow);
    for (int layer = 0; layer < N_LAYER; ++layer) {
        adam_update_param(&model->attn_wq[layer], step, total_steps, beta1_pow, beta2_pow);
        adam_update_param(&model->attn_wk[layer], step, total_steps, beta1_pow, beta2_pow);
        adam_update_param(&model->attn_wv[layer], step, total_steps, beta1_pow, beta2_pow);
        adam_update_param(&model->attn_wo[layer], step, total_steps, beta1_pow, beta2_pow);
        adam_update_param(&model->mlp_fc1[layer], step, total_steps, beta1_pow, beta2_pow);
        adam_update_param(&model->mlp_fc2[layer], step, total_steps, beta1_pow, beta2_pow);
    }
}

static int sample_from_probs(const float *probs, int n) {
    float r = rand_uniform();
    float cumulative = 0.0f;
    for (int i = 0; i < n; ++i) {
        cumulative += probs[i];
        if (r <= cumulative) {
            return i;
        }
    }
    return n - 1;
}

static void generate_samples(Model *model, TrainCache *cache, const Dataset *dataset, int num_samples, float temperature) {
    float *host_probs = (float *)xmalloc((size_t)dataset->vocab_size * sizeof(float));
    printf("\n--- inference (new, hallucinated names) ---\n");
    for (int sample_idx = 0; sample_idx < num_samples; ++sample_idx) {
        int token_id = dataset->bos_id;
        char sample[BLOCK_SIZE + 1];
        int length = 0;

        for (int pos = 0; pos < BLOCK_SIZE; ++pos) {
            forward_position(model, cache, dataset->vocab_size, token_id, pos);
            PositionCache *pc = &cache->pos[pos];
            launch_scaled_softmax(pc->logits, pc->probs, 1.0f / temperature, dataset->vocab_size);
            copy_to_host(host_probs, pc->probs, dataset->vocab_size);
            token_id = sample_from_probs(host_probs, dataset->vocab_size);
            if (token_id == dataset->bos_id) {
                break;
            }
            sample[length++] = (char)dataset->id_to_char[token_id];
        }

        sample[length] = '\0';
        printf("sample %2d: %s\n", sample_idx + 1, sample);
    }
    free(host_probs);
}

static Options parse_options(int argc, char **argv) {
    Options opts;
    opts.num_steps = 1000;
    opts.num_samples = 20;
    opts.temperature = 0.5f;
    opts.input_path = "input.txt";

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            opts.num_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--samples") == 0 && i + 1 < argc) {
            opts.num_samples = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
            opts.temperature = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            opts.input_path = argv[++i];
        } else {
            fprintf(stderr, "usage: %s [--steps N] [--samples N] [--temperature T] [--input PATH]\n", argv[0]);
            exit(1);
        }
    }

    if (opts.num_steps <= 0) {
        fprintf(stderr, "--steps must be positive\n");
        exit(1);
    }
    if (opts.num_samples < 0) {
        fprintf(stderr, "--samples must be non-negative\n");
        exit(1);
    }
    if (opts.temperature <= 0.0f) {
        fprintf(stderr, "--temperature must be > 0\n");
        exit(1);
    }

    return opts;
}

int main(int argc, char **argv) {
    srand(42);
    Options opts = parse_options(argc, argv);

    int device_count = 0;
    cudaError_t device_err = cudaGetDeviceCount(&device_count);
    if (device_err != cudaSuccess || device_count == 0) {
        fprintf(stderr, "microgpt.cu requires a CUDA-capable GPU at runtime");
        if (device_err != cudaSuccess) {
            fprintf(stderr, " (%s)", cudaGetErrorString(device_err));
        }
        fprintf(stderr, "\n");
        return 1;
    }

    CHECK_CUBLAS(cublasCreate(&g_cublas));

    Dataset dataset;
    load_dataset(&dataset, opts.input_path);
    printf("num docs: %d\n", dataset.docs.count);
    printf("vocab size: %d\n", dataset.vocab_size);

    FloatArena param_arena;
    init_arena(&param_arena, param_storage_count(dataset.vocab_size));

    Model model;
    init_model(&model, &param_arena, dataset.vocab_size);
    printf("num params: %zu\n", total_params(&model));

    FloatArena cache_arena;
    init_arena(&cache_arena, train_cache_storage_count(dataset.vocab_size));

    TrainCache cache;
    init_train_cache(&cache, dataset.vocab_size, &cache_arena);

    FloatArena scratch_arena;
    init_arena(&scratch_arena, scratch_storage_count(dataset.vocab_size));

    Scratch scratch;
    init_scratch(&scratch, dataset.vocab_size, &scratch_arena);

    int tokens[BLOCK_SIZE + 2];
    for (int step = 0; step < opts.num_steps; ++step) {
        const char *doc = dataset.docs.items[step % dataset.docs.count];
        int token_count = tokenize_doc(&dataset, doc, tokens);
        int n = token_count - 1;
        if (n > BLOCK_SIZE) {
            n = BLOCK_SIZE;
        }

        float loss = forward_sequence(&model, &cache, &dataset, tokens, n);
        backward_sequence(&model, &cache, &scratch, &dataset, n);
        adam_update_model(&model, step, opts.num_steps);

        printf("step %4d / %4d | loss %.4f\r", step + 1, opts.num_steps, loss);
        fflush(stdout);
    }

    generate_samples(&model, &cache, &dataset, opts.num_samples, opts.temperature);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUBLAS(cublasDestroy(g_cublas));
    return 0;
}
@hmxiong
Comment
 
Leave a comment
Footer
© 2026 GitHub, Inc.
Footer navigation
Terms
Privacy
Security
Status
Community
Docs
Contact
Manage cookies
Do not share my personal information
