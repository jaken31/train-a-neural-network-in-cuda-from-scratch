"""
Train a Neural Network in CUDA from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - report_cuda_error
#include <cstdio>
#include <cuda_runtime.h>

int report_cuda_error(cudaError_t err, const char* what) {
    // TODO: return 0 on cudaSuccess; otherwise print
    // "CUDA error in <what>: <cudaGetErrorString(err)>" to stderr and return 1
    if (err == cudaSuccess) {
        return 0;
    }
    fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(err));
    return 1;
}

# Step 2 - device_upload
#include <cuda_runtime.h>

float* device_upload(const float* host, int n) {
    // TODO: cudaMalloc n 2floats, cudaMemcpy host -> device, return the device pointer
    size_t size = n * sizeof(float);
    float* h_ptr;
    cudaMalloc(&h_ptr, size);
    cudaMemcpy(h_ptr, host, size, cudaMemcpyHostToDevice);
    return h_ptr;
}

int* device_upload(const int* host, int n) {
    // TODO: same for ints
    size_t size = n * sizeof(int);
    int* h_ptr;
    cudaMalloc(&h_ptr, size);
    cudaMemcpy(h_ptr, host, size, cudaMemcpyHostToDevice);
    return h_ptr;
}

void device_download(float* host, const float* dev, int n) {
    // TODO: cudaMemcpy device -> host
    size_t size = n * sizeof(float);
    cudaMemcpy(host, dev, size, cudaMemcpyDeviceToHost);
}

# Step 3 - make_spiral
#include <cmath>

float lcg_uniform(unsigned* state) {
    // TODO: *state = *state * 1664525u + 1013904223u; return (*state >> 8) / 16777216.0f
    *state = *state * 1664525u + 1013904223u;
    return (*state >> 8) / 16777216.0f;
}

float lcg_normal(unsigned* state) {
    // TODO: Box-Muller from two lcg_uniform draws (u1 first, then u2)
    float u1 = lcg_uniform(state);
    float u2 = lcg_uniform(state);
    return sqrtf(-2.0f * logf(u1 + 1e-7f)) * cosf(6.2831853f * u2);
}

void make_spiral(float* X, int* y, int points_per_class, int classes, float noise, unsigned seed) {
    // TODO: one generator seeded with `seed`; for each class c, for each point i:
    // r = i / (points_per_class - 1), t = 4c + 4r + noise * normal, X = (r sin t, r cos t), y = c
    unsigned state = seed;
    for (int c = 0; c < classes; c++) {
        for (int i = 0; i < points_per_class; i++) {
            int k = c * points_per_class + i;
            float r = (float) i / (float)(points_per_class - 1);
            float t = 4.0f * c + 4.0f * r + noise * lcg_normal(&state);
            X[2 * k] = r * sinf(t);
            X[2*k + 1] = r * cosf(t);
            y[k] = c;
        }
    }
}

# Step 4 - matmul_kernel
#include <cuda_runtime.h>
#define BLOCK 16
__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // TODO: col from x, row from y, guard, accumulate over k, store C[row * N + col]
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    } 
}

void launch_matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    // TODO: 16 x 16 blocks, grid covering N columns (x) and M rows (y)
    dim3 block(BLOCK, BLOCK);
    dim3 grid((N - 1 + BLOCK) / BLOCK, (M - 1 + BLOCK) / BLOCK);
    matmul_kernel<<<grid, block>>>(A, B, C, M, N, K);
}

# Step 5 - add_bias_kernel
#include <cuda_runtime.h>

__global__ void add_bias_kernel(float* Z, const float* b, int rows, int cols) {
    // TODO: one thread per element; element i gets b[i % cols]
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    Z[i] += b[i % cols];
}

void launch_add_bias(float* Z, const float* b, int rows, int cols) {
    // TODO: 256 threads per block over rows * cols elements
    add_bias_kernel<<<(rows * cols + 255) / 256, 256>>>(Z, b, rows, cols);
}

# Step 6 - relu_kernel
#include <cuda_runtime.h>

__global__ void relu_kernel(float* Z, int n) {
    // TODO: Z[i] = max(Z[i], 0) for i < n
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        Z[i] = max(Z[i], 0.0f);
    }
}

void launch_relu(float* Z, int n) {
    // TODO: 256 threads per block
    relu_kernel<<<(n + 255) / 256, 256>>>(Z, n);
}

# Step 7 - softmax_rows_kernel
#include <cuda_runtime.h>

__global__ void softmax_rows_kernel(const float* logits, float* probs, int rows, int cols) {
    // TODO: one thread per row: max, sum of exp(z - max), then normalise
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) {
        return;
    }
    float row_max = -FLT_MIN;
    for (int j = 0; j < cols; j++) {
        row_max = max(logits[i * cols + j], row_max);
    }
    float sum = 0.0f;
    for (int s = 0; s < cols; s++) {
        sum += expf(logits[i * cols + s] - row_max);
    }
    for (int k = 0; k < cols; k++) {
        probs[i * cols + k] = expf(logits[i * cols + k] - row_max)/sum;
    }
}

void launch_softmax_rows(const float* logits, float* probs, int rows, int cols) {
    // TODO: 128 threads per block, one thread per row
    softmax_rows_kernel<<<(rows + 127) / 128, 128>>>(logits, probs, rows, cols);
}

# Step 8 - cross_entropy_kernel
#include <cmath>
#include <cuda_runtime.h>

__global__ void cross_entropy_kernel(const float* probs, const int* labels, int rows, int cols, float* total) {
    // TODO: per-thread -log(max(p[label], 1e-12)), shared-memory tree sum of 256 values,
    // thread 0 atomicAdds the block sum into *total
    __shared__ float sdata[256];
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    sdata[tid] = (r < rows) ? -logf(fmaxf(probs[r * cols + labels[r]], 1e-12f)) : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        sdata[tid] += tid < s ? sdata[tid + s] : 0.0f;
        __syncthreads();
    }
    if (tid == 0) atomicAdd(total, sdata[0]);
}

float cross_entropy_loss(const float* probs, const int* labels, int rows, int cols) {
    // TODO: zeroed device float, launch 256-thread blocks, copy back, free, return total / rows
    float* t_ptr;
    float total = 0.0f;
    cudaMalloc(&t_ptr, sizeof(float));
    cudaMemset(t_ptr, 0, sizeof(float));
    cross_entropy_kernel<<<(rows + 255) / 256, 256>>>(probs, labels, rows, cols, t_ptr);
    cudaMemcpy(&total, t_ptr, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(t_ptr);
    return total / rows;
}

# Step 9 - init_mlp
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

struct MLP {
    int in_dim, hidden, out_dim;
    float *W1, *b1, *W2, *b2;   // device pointers
};

MLP init_mlp(int in_dim, int hidden, int out_dim, unsigned seed) {
    MLP m = {in_dim, hidden, out_dim, nullptr, nullptr, nullptr, nullptr};
    // TODO: one generator seeded with `seed`; W1 then W2, (2u - 1) * sqrt(6 / fan_in); zero biases;
    // upload all four with device_upload
    unsigned state = seed;
    std::vector<float> host_W1(hidden * in_dim);
    for (int i = 0; i < hidden * in_dim; i++) {
        host_W1[i] = (2 * lcg_uniform(&state) - 1) * sqrtf(6.0f / in_dim);
    }
    std::vector<float> host_W2(hidden * out_dim);
    for (int i = 0; i < hidden * out_dim; i++) {
        host_W2[i] = (2 * lcg_uniform(&state) - 1) * sqrtf(6.0f / hidden);
    }
    std::vector<float> host_b1(hidden);
    for (int i = 0; i < hidden; i++) {
        host_b1[i] = 0.0f;
    }
    std::vector<float> host_b2(out_dim);
    for (int i = 0; i < out_dim; i++) {
        host_b2[i] = 0.0f;
    }
    m.W1 = device_upload(host_W1.data(), hidden * in_dim);
    m.W2 = device_upload(host_W2.data(), hidden * out_dim);
    m.b1 = device_upload(host_b1.data(), hidden);
    m.b2 = device_upload(host_b2.data(), out_dim);
    return m;
}

void free_mlp(MLP& m) {
    cudaFree(m.W1);
    m.W1 = nullptr;
    cudaFree(m.W2);
    m.W2 = nullptr;
    cudaFree(m.b1);
    m.b1 = nullptr;
    cudaFree(m.b2);
    m.b2 = nullptr;
}

# Step 10 - alloc_workspace
#include <cuda_runtime.h>

struct Workspace {
    int batch;
    float *H, *P;                    // forward: hidden activations, probabilities
    float *dZ2, *dH;                 // backward: gradients w.r.t. logits and hidden
    float *dW1, *db1, *dW2, *db2;    // parameter gradients
};

Workspace alloc_workspace(const MLP& m, int batch) {
    Workspace w = {batch, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    // TODO: cudaMalloc and cudaMemset all eight buffers with the sizes in the description
    cudaMalloc(&w.H, (size_t) batch * m.hidden * sizeof(float));
    cudaMalloc(&w.dH, (size_t) batch * m.hidden * sizeof(float));
    cudaMalloc(&w.P, (size_t) batch * m.out_dim * sizeof(float));
    cudaMalloc(&w.dZ2, (size_t) batch * m.out_dim * sizeof(float));
    cudaMalloc(&w.dW1, (size_t) m.in_dim * m.hidden * sizeof(float));
    cudaMalloc(&w.db1, (size_t) m.hidden * sizeof(float));
    cudaMalloc(&w.dW2, (size_t) m.hidden * m.out_dim * sizeof(float));
    cudaMalloc(&w.db2, (size_t) m.out_dim * sizeof(float));

    cudaMemset(w.H, 0,  (size_t) batch * m.hidden * sizeof(float));
    cudaMemset(w.dH, 0,  (size_t) batch * m.hidden * sizeof(float));
    cudaMemset(w.P,   0,(size_t) batch * m.out_dim * sizeof(float));
    cudaMemset(w.dZ2, 0, (size_t) batch * m.out_dim * sizeof(float));
    cudaMemset(w.dW1, 0, (size_t) m.in_dim * m.hidden * sizeof(float));
    cudaMemset(w.db1, 0, (size_t) m.hidden * sizeof(float));
    cudaMemset(w.dW2, 0, (size_t) m.hidden * m.out_dim * sizeof(float));
    cudaMemset(w.db2, 0, (size_t) m.out_dim * sizeof(float));
    return w;
}

void free_workspace(Workspace& w) {
    // TODO: cudaFree all eight, set them to nullptr
    cudaFree(w.H); w.H = nullptr;
    cudaFree(w.dH); w.dH = nullptr;
    cudaFree(w.P); w.P = nullptr;
    cudaFree(w.dZ2); w.dZ2 = nullptr;
    cudaFree(w.dW1); w.dW1 = nullptr;
    cudaFree(w.db1); w.db1 = nullptr;
    cudaFree(w.dW2); w.dW2 = nullptr;
    cudaFree(w.db2); w.db2 = nullptr;
}

# Step 11 - forward_mlp
void forward_mlp(const MLP& m, const float* X, int batch, Workspace& w) {
    // TODO: H = relu(X @ W1 + b1); P = softmax(H @ W2 + b2), using the earlier launchers
    launch_matmul(X, m.W1, w.H, batch, m.hidden, m.in_dim);
    launch_add_bias(w.H, m.b1, batch, m.hidden);
    launch_relu(w.H, batch * m.hidden);

    launch_matmul(w.H, m.W2, w.P, batch, m.out_dim , m.hidden);
    launch_add_bias(w.P, m.b2, batch, m.out_dim);
    launch_softmax_rows(w.P, w.P, batch, m.out_dim);
}

# Step 12 - softmax_ce_grad_kernel
#include <cuda_runtime.h>

__global__ void softmax_ce_grad_kernel(const float* P, const int* labels, float* dZ, int rows, int cols) {
    // TODO: dZ[r][j] = (P[r][j] - [j == labels[r]]) / rows
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (rows * cols < idx) {
        return;
    }
    int r = idx / cols;
    int j = idx % cols;
    dZ[r * cols + j] = (P[r * cols + j] - (j == labels[r] ? 1 : 0)) / rows;
}

void launch_softmax_ce_grad(const float* P, const int* labels, float* dZ, int rows, int cols) {
    // TODO: 256 threads per block over rows * cols
    softmax_ce_grad_kernel<<<(rows * cols + 255) / 256, 256>>>(P, labels, dZ, rows, cols);
}

# Step 13 - matmul_tn_kernel (not yet solved)
# TODO: implement

# Step 14 - matmul_nt_kernel (not yet solved)
# TODO: implement

# Step 15 - column_sum_kernel (not yet solved)
# TODO: implement

# Step 16 - relu_backward_kernel (not yet solved)
# TODO: implement

# Step 17 - backward_mlp (not yet solved)
# TODO: implement

# Step 18 - sgd_update_kernel (not yet solved)
# TODO: implement

# Step 19 - train_step (not yet solved)
# TODO: implement

# Step 20 - train_mlp (not yet solved)
# TODO: implement

# Step 21 - accuracy_kernel (not yet solved)
# TODO: implement

# Step 22 - time_train_step_ms (not yet solved)
# TODO: implement

