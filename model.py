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

# Step 8 - cross_entropy_kernel (not yet solved)
# TODO: implement

# Step 9 - init_mlp (not yet solved)
# TODO: implement

# Step 10 - alloc_workspace (not yet solved)
# TODO: implement

# Step 11 - forward_mlp (not yet solved)
# TODO: implement

# Step 12 - softmax_ce_grad_kernel (not yet solved)
# TODO: implement

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

