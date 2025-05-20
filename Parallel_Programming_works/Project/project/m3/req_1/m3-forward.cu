#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#include <mma.h>
using namespace nvcuda;

#define BLOCK_SIZE 256
#define TILE_WIDTH 16
#define MATMUL_TILE_WIDTH 16

#define M 16
#define N 16
#define TF_K 8

__global__ void matmul_conv_fused(const float *mask, const float *input, float *output,
                                  int Batch, int Map_out, int Channel, int Height, int Width, int K)
{
    __shared__ float tileA[2*M][TF_K];
    __shared__ float tileB[TF_K][2*N];
    __shared__ float tileC[2*M][2*N];

    const int Width_out = (Width - K) + 1;
    const int Height_out = Height - K + 1;
 
    // Initialize the output fragment
    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    int by = blockIdx.y, bx = blockIdx.x, ty = threadIdx.y, tx = threadIdx.x;
    int bz = blockIdx.z;

    int row = by * TILE_WIDTH + ty, col = bx * TILE_WIDTH + tx;
    float val = 0;

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]

    // The unrolled matrix is of size (Channel * K * K) * (Height_out * Width_out)
    // The unrolled mask is of size (Map_out) * (Channel * K * K)
    int input_unrolled_height = Channel * K * K;
    int input_unrolled_width = Height_out * Width_out;
    int mask_unrolled_height = Map_out;
    int mask_unrolled_width = Channel * K * K;
    int inputC_height = Map_out;
    int inputC_width = Height_out * Width_out;

    // get the row and column index of the block in warp
    int Row_W = (threadIdx.x / 32) / 2;
    int Col_W = (threadIdx.x / 32) % 2;

    // get the row and column index of the block in the grid
    int Row_G = 2 * M * by;
    int Col_G = 2 * N * bx;
    
    // load data into shared memory
    for (int i = 0; i < (mask_unrolled_width + TF_K - 1) / TF_K; i++){ 

        //Load the mask into shared memory
        int col_m = i * TILE_WIDTH + tx;
        int idx = 2*tx;
        for (int j = 0; j < 2; j++){
            int Row = (idx + j) / TF_K;
            int Col = (idx + j) % TF_K;
            int col_m = TF_K * i + Col;
            int row_m = Row_G + Row;
            if ((row_m < mask_unrolled_height) && (col_m < mask_unrolled_width)){           
                tileA[Row][Col] = wmma::__float_to_tf32(mask[row_m * mask_unrolled_width + col_m]);
            } else { 
                tileA[Row][Col] = wmma::__float_to_tf32(0.0f);
            }
        }
        __syncthreads();

        // Load the input into shared memory
        int idxA = 2*tx;
        for (int j = 0; j < 2; j++){
            int Row = (idxA + j) / (2*N);
            int Col = (idxA + j) % (2*N);
            int col_i = Col_G + Col;
            int row_i = i*TF_K + Row;
            if ((row_i < input_unrolled_height) && (col_i < input_unrolled_width)) {
                int c = row_i / (K * K); 
                
                int h = col_i / Width_out;
                int p = (row_i - c * K * K) / K;

                int w = col_i % Width_out;
                int q = (row_i - c * K * K) % K;
                tileB[Row][Col] = wmma::__float_to_tf32(in_4d(bz, c, h + p, w + q));
            }
            else {
                tileB[Row][Col] = wmma::__float_to_tf32(0.0f);
            }
        }
        __syncthreads();

        wmma::load_matrix_sync(a_frag, &tileA[M * Row_W][0], TF_K);
        wmma::load_matrix_sync(b_frag, &tileB[0][N * Col_W], 2 * N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
    }

    wmma::store_matrix_sync(&tileC[M * Row_W][N * Col_W], c_frag, 2 * N, wmma::mem_row_major);
    __syncthreads();

    //     if (row < mask_unrolled_height && col < input_unrolled_width){ 
    //         for (int j = 0; j < TILE_WIDTH; j++){
    //             val += tileB[ty][j] * tileA[j][tx];
    //         }
    //     }
    //     __syncthreads(); 
    // }

    // store the result in the output matrix
    int idx = 8*tx;
    for (int j = 0; j < 8; j++){
        int Row = (idx + j) / (2*N);
        int Col = (idx + j) % (2*N);
        int col_o = Col_G + Col;
        int row_o = Row_G + Row;
        if ((row_o < mask_unrolled_height) && (col_o < input_unrolled_width)) {
            out_4d(bz, row_o, col_o/Width_out, col_o%Width_out) = tileC[Row][Col];
        }
    }

    // if (row < mask_unrolled_height && col < input_unrolled_width){
    //     // Calculate the output index
    //     output[bz * (mask_unrolled_height * input_unrolled_width) + row * input_unrolled_width + col] = val; 
    // }

    #undef out_4d
    #undef in_4d
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Allocate memory and copy over the relevant data structures to the GPU

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    cudaMalloc((void **)device_input_ptr, Batch * Channel * Height * Width * sizeof(float));
    cudaMalloc((void **)device_mask_ptr, Map_out * Channel * K * K * sizeof(float));
    cudaMalloc((void **)device_output_ptr, Batch * Map_out * Height_out * Width_out * sizeof(float));

    cudaMemcpy(*device_input_ptr, host_input, Batch * Channel * Height * Width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, Map_out * Channel * K * K * sizeof(float), cudaMemcpyHostToDevice);
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Set the kernel dimensions and call the fused kernel
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    // const int Height_unrolled = Channel * K * K;
    // const int Width_unrolled = Batch * Height_out * Width_out;

    // float *unrolled_matrix;  // Pointer to device memory for storing the unrolled matrix
    // float *matmul_output;    // Pointer to device memory for storing the result of matrix multiplication
    // cudaMalloc((void**)&unrolled_matrix, (size_t) Height_unrolled * Width_unrolled * sizeof(float));
    // cudaMalloc((void**)&matmul_output, (Batch * Map_out * Height_out * Width_out) * sizeof(float));

    dim3 gridDim((Height_out * Width_out + N * 2 - 1) / (N * 2), (Map_out + M * 2 - 1) / (M * 2), Batch);
    dim3 blockDim(128, 1, 1);
    matmul_conv_fused<<<gridDim,blockDim>>> (device_mask, device_input, device_output,
                                            Batch, Map_out, Channel, Height, Width, K);
//     cudaDeviceSynchronize();

//     cudaFree(matmul_output);
//     cudaFree(unrolled_matrix);
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Copy the output back to host

    // TODO: Free device memory
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    cudaMemcpy(host_output, device_output, Batch * Map_out * Height_out * Width_out * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);

}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}