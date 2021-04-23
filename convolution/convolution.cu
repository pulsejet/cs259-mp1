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

#include "dnn.hpp"

//Define the parameters if not defined externally
#ifndef Sy
  #define Sy 1
  #define Sx 1
#endif

/*
#ifndef Ky
  #define Nx 224
  #define Ny 224
  #define Ky 3
  #define Kx 3
  #define Nn 64
  #define Ni 64
#endif
*/

#ifndef Tnn
  //Tiling Sizes
  #define Tnn 32
  #define Tn  16
  #define Ti  16
  //#define Ty  4
  //#define Tx  4
#endif

#define NYPAD (Ny+Ky)
#define NXPAD (Nx+Kx)

#define NYSCL (Ny/Sy)
#define NXSCL (Nx/Sx)

#define SYNAPSE_SIZE (1L*Ky*Kx*Nn*Ni)

VTYPE (*synapse)[Ky][Kx][Nn][Ni];
VTYPE (*neuron_i)[NYPAD][NXPAD][Ni];
VTYPE (*neuron_n)[NYSCL][NXSCL][Nn];
VTYPE (*neuron_n2)[NYSCL][NXSCL][Nn];

template<class T>
void printMatrix(const dim3 &dim, const std::vector<T> &vec) {
    for(int i=0; i < dim.y; i++)  {
        for(int j=0; j < dim.x; j++)
            std::cout << vec[i * dim.x + j] << ' ';
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void fill_convolution_shared_simple(VTYPE (&synapse)[Ky][Kx][Nn][Ni],
                                    VTYPE (&neuron_i)[NYPAD][NXPAD][Ni]) {
    for(int yy = 0; yy < Ky; ++yy) {
        for(int xx = 0; xx < Kx; ++xx) {
            for(int nn = 0; nn < Nn; ++nn) {
                for(int ni = 0; ni < Ni; ++ni) {
                    synapse[yy][xx][nn][ni] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
                } } } }
    for(int yy = 0; yy < NYPAD; ++yy) {
        for(int xx = 0; xx < NXPAD; ++xx) {
            for(int ni = 0; ni < Ni; ++ni) {
                neuron_i[yy][xx][ni] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
            }  }  }
}

void  convolution_layer(VTYPE (&synapse)[Ky][Kx][Nn][Ni],
                               VTYPE (&neuron_i)[NYPAD][NXPAD][Ni],
                               VTYPE (&neuron_n)[NYSCL][NXSCL][Nn]) {
  VTYPE sum[Nn]={0};

  // — Original code — (excluding nn, ii loops)
  int yout = 0;
  for (int y = 0; y < Ny; y += Sy) { // tiling for y;
    int xout = 0;
    for (int x = 0; x < Ny; x += Sx) { // tiling for x;
      for (int nn = 0; nn < Nn; nn += Tn) {
        for (int n = nn; n < nn + Tn; n++) {
          sum[n]=0;
        }

        // sliding window;
        for (int ky = 0; ky < Ky; ky++)
          for (int kx = 0; kx < Kx; kx++)
            for (int n = nn; n < nn + Tn; n++)
              for (int i = 0; i < Ni; i++) {
                VTYPE sv = synapse[ky][kx][n][i];
                VTYPE nv = neuron_i[ky + y][kx + x][i];
                sum[n]+=sv*nv;
              }
        for (int n = nn; n < nn + Tn; n++) {
          neuron_n[yout][xout][n] = transfer(sum[n]);
        }
      }
      xout++;
    }
    yout++;
  }
}

// Is this a leaky relu?
__device__ float transfer_gpu(VTYPE i) {
    return (i>0) ? i : i/4;
}

// GPU FUNCTION
__global__ void convolution3d(float *d_synapse, float *d_neuron_i, float *d_neuron_n) {
    int destX = blockIdx.x * blockDim.x + threadIdx.x;
    int destY = blockIdx.y * blockDim.y + threadIdx.y;
    int destZ = blockIdx.z * blockDim.z + threadIdx.z;

    __shared__ float s_mem[Ty+Ky-1][Tx+Kx-1][Tn];

    int destCoord = destY * Ni * Nx + destX * Ni + destZ;

    s_mem[threadIdx.y][threadIdx.x][threadIdx.z] = d_neuron_i[destY*NXPAD*Ni + destX*Ni + destZ];

    __syncthreads();

    float sum = 0;
    for (int ky = 0; ky < Ky; ky++)
        for (int kx = 0; kx < Kx; kx++)
            for (int i = 0; i < Ni; i++) {
                float sv = d_synapse[ky*Kx*Nn*Ni + kx*Nn*Ni + destZ*Ni + i];
                float nv = 0;

                if (threadIdx.x + kx >= Tx || threadIdx.y + ky >= Ty ||
                    i < blockIdx.z * Tn || i >= (blockIdx.z + 1) * Tn)

                    nv = d_neuron_i[(ky + destY)*NXPAD*Ni + (kx + destX)*Ni + i];
                else
                    nv = s_mem[ky + threadIdx.y][kx + threadIdx.x][i%Tn];

                sum += sv*nv;
            }

    d_neuron_n[destCoord] = transfer_gpu(sum);
    __syncthreads();
}

/**
 * Program main
 */
int main(int argc, char **argv) {
    size_t synapse_size = SYNAPSE_SIZE*sizeof(VTYPE);
    size_t neuron_i_size = NYPAD*NXPAD*Ni*sizeof(VTYPE);
    size_t neuron_n_size = NYSCL*NXSCL*Nn*sizeof(VTYPE);

    synapse   = (VTYPE (*)[Ky][Kx][Nn][Ni])   aligned_malloc(64, synapse_size);
    neuron_i  = (VTYPE (*)[NYPAD][NXPAD][Ni]) aligned_malloc(64, neuron_i_size);
    neuron_n  = (VTYPE (*)[NYSCL][NXSCL][Nn]) aligned_malloc(64, neuron_n_size);

    fill_convolution_shared_simple(*synapse,*neuron_i);

    std::cout << "initializing arrays\n";

    float *d_synapse;
    float *d_neuron_i;
    float *d_neuron_n;

    checkCudaErrors(cudaMalloc(&d_synapse, synapse_size));
    checkCudaErrors(cudaMalloc(&d_neuron_i, neuron_i_size));
    checkCudaErrors(cudaMalloc(&d_neuron_n, neuron_n_size));

    // copy host memory to device
    checkCudaErrors(cudaMemcpy(d_synapse, synapse, synapse_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_neuron_i, neuron_i, neuron_i_size, cudaMemcpyHostToDevice));

    dim3 block_dim(Tx, Ty, Tn);
    dim3 grid_dim((Nx + Tx - 1) / Tx, (Ny + Ty - 1) / Ty, (Ni + Tn - 1) / Tn);

    convolution3d<<<grid_dim, block_dim>>>(d_synapse, d_neuron_i, d_neuron_n);
    gpuErrchk(cudaPeekAtLastError());

    cudaDeviceSynchronize();

    std::vector<float> retval(neuron_n_size / sizeof(VTYPE));
    checkCudaErrors(cudaMemcpy(retval.data(), d_neuron_n, retval.size() * sizeof(float), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_synapse));
    checkCudaErrors(cudaFree(d_neuron_i));
    checkCudaErrors(cudaFree(d_neuron_n));

    for (int i = 0; i < 100; i++) {
        // std::cout << ((float*)synapse)[i] << " ";
        std::cout << retval[i] << " ";
    }
    std::cout << std::endl;

    convolution_layer(*synapse,*neuron_i,*neuron_n);

    for (int i = 0; i < retval.size(); i++) {
        float g = ((float*)neuron_n)[i];
        float c = retval[i];
        if (g < c - 0.1f || g > c + 0.1f) {
            std::cout << c << " != " << g << " index: " << i;
            exit(0);
        }
    }
    std::cout << "VALIDATION SUCCESSFUL" << std::endl;

    return 0;
}

