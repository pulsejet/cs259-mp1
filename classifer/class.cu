/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication which makes use of shared memory
 * to ensure data reuse, the matrix multiplication is done using tiling approach.
 * It has been written for clarity of exposition to illustrate various CUDA programming
 * principles, not with the goal of providing the most performant generic kernel for matrix multiplication.
 * See also:
 * V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
 * in Proc. 2008 ACM/IEEE Conf. on Supercomputing (SC '08),
 * Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
 */

// System includes
#include <stdio.h>
#include <assert.h>
#include <vector>
#include <iostream>
#include <utility>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

template<class T>
void printMatrix(const dim3 &dim, const std::vector<T> &vec) {
    for(int i=0; i < dim.y; i++)  {
        for(int j=0; j < dim.x; j++)
            std::cout << vec[i * dim.x + j] << ' ';
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */
template <int BLOCK_SIZE> __global__ void MatrixMulCUDA(float *C, float *A,
                                                        float *B, int wA,
                                                        int wB) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
            a <= aEnd;
            a += aStep, b += bStep) {
        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

void ConstantInit(float *data, int size, float val) {
    for (int i = 0; i < size; ++i) {
        data[i] = val;
    }
}

/**
 * Run a simple test of matrix multiplication using CUDA
 */
std::pair<dim3, std::vector<float>>
MatrixMultiply(int block_size,
               const dim3 &dimsA, const dim3 &dimsB,
               const std::vector<float> &h_A, std::vector<float> &h_B,
               bool verify = false) {

    // Allocate device memory
    float *d_A, *d_B, *d_C;

    // Allocate host matrix C
    dim3 dimsC(dimsB.x, dimsA.y, 1);
    unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);

    checkCudaErrors(cudaMalloc(&d_A, h_A.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_B, h_B.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_C), mem_size_C));

    // copy host memory to device
    checkCudaErrors(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(float), cudaMemcpyHostToDevice));

    // Setup execution parameters
    dim3 threads(block_size, block_size);
    dim3 grid((dimsB.x + threads.x - 1) / threads.x, (dimsA.y + threads.y - 1) / threads.y);

    // Create and start timer
    printf("Computing result using CUDA Kernel...\n");

    // Performs warmup operation using matrixMul CUDA kernel
    MatrixMulCUDA<16> <<< grid, threads >>>(d_C, d_A, d_B, dimsA.x, dimsB.x);

    printf("done\n");

    cudaDeviceSynchronize();

    std::vector<float> retval(dimsC.x * dimsC.y);
    checkCudaErrors(cudaMemcpy(retval.data(), d_C, retval.size() * sizeof(float), cudaMemcpyDeviceToHost));

    if (verify) {
        for (size_t i = 0; i < dimsA.y; i++) {
            for (size_t j = 0; j < dimsB.x; j++) {
                float cij = 0;
                for (size_t k = 0; k < dimsA.x; k++) {
                    cij += h_A[i * dimsA.x + k] * h_B[k * dimsB.x + j];
                }

                float gpu_val = retval[i * dimsC.x + j];
                if (cij > gpu_val + 1.0f || cij < gpu_val - 1.0f) {
                    std::cerr << "VER_ERROR==" << cij << " " << retval[i * dimsC.x + j] << std::endl;
                }
            }
        }

        printf("Verification complete\n");
    }

    // Clean up memory
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));

    return std::pair<dim3, std::vector<float>>(dimsC, retval);
}

/**
 * Program main
 */
int main(int argc, char **argv) {
    int block_size = 16;

    dim3 dimsX(Ni, BATCH_SIZE);
    dim3 dimsW(Nn, Ni);

    // host
    std::vector<float> h_X(dimsX.x * dimsX.y);
    std::vector<float> h_W(dimsW.x * dimsW.y);

    // init
    generate(h_W.begin(), h_W.end(), []() { return ((float) (rand() % 100)) / 100.0; });
    generate(h_X.begin(), h_X.end(), []() { return (float) (rand() % 100); });
    auto result = MatrixMultiply(block_size, dimsX, dimsW, h_X, h_W, false);

    printf("MatrixX(%d,%d), MatrixW%d,%d), MatrixY(%d,%d)\n",
           dimsX.x, dimsX.y, dimsW.x, dimsW.y, result.first.x, result.first.y);

    return 0;
}

