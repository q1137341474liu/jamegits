#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define BLOCK_SIZE 256
#define TILE_WIDTH 16
#define MATMUL_TILE_WIDTH 16

__constant__ float constant_mask [10000];

__global__ void matmul_conv_fused(const float *mask, const float *input, float *output,
                                  int Batch, int Map_out, int Channel, int Height, int Width, int K)
{
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];


    const int Width_out = (Width - K) + 1;
    const int Height_out = Height - K + 1;
    
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

    // load data into shared memory
    for (int i = 0; i < (mask_unrolled_width - 1) / TILE_WIDTH + 1; i++){ 

        // Load the mask into shared memory
        int col_m = i * TILE_WIDTH + tx;
        if ((row < mask_unrolled_height) && (col_m < mask_unrolled_width)){           
            tileB[ty][tx] = constant_mask[row * mask_unrolled_width + col_m];
        } else { 
            tileB[ty][tx] = 0;
        }
        // __syncthreads();

        // Load the input into shared memory
        if ((i * TILE_WIDTH + ty < input_unrolled_height) && (col < input_unrolled_width)) {
            int c = (i * TILE_WIDTH + ty) / (K * K); 
            
            int h = col / Width_out;
            int p = (i * TILE_WIDTH + ty - c * K * K) / K;

            int w = col % Width_out;
            int q = (i * TILE_WIDTH + ty - c * K * K) % K;

            if ((h + p < Height) && (w + q < Width)){
                tileA[ty][tx] = in_4d(bz, c, h + p, w + q);
            } else {
                tileA[ty][tx] = 0;
            }
        } else {

            tileA[ty][tx] = 0;
        }
        __syncthreads();

        if (row < mask_unrolled_height && col < input_unrolled_width){ 
            for (int j = 0; j < TILE_WIDTH; j++){
                val += tileB[ty][j] * tileA[j][tx];
            }
        }
        __syncthreads(); 
    }

    if (row < mask_unrolled_height && col < input_unrolled_width){
        // Calculate the output index
        output[bz * (mask_unrolled_height * input_unrolled_width) + row * input_unrolled_width + col] = val; 
    }

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
    cudaMemcpyToSymbol(constant_mask, host_mask, Map_out * Channel * K * K * sizeof(float));
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

    dim3 grid_unroll(ceil((1.0 * Height_out * Width_out) / TILE_WIDTH), ceil((1.0 * Map_out) / TILE_WIDTH), Batch);
    dim3 block_unroll(TILE_WIDTH, TILE_WIDTH, 1);
    matmul_conv_fused<<<grid_unroll,block_unroll>>> (device_mask, device_input, device_output,
                                            Batch, Map_out, Channel, Height, Width, K);
    // cudaDeviceSynchronize();

    // cudaFree(matmul_output);
    // cudaFree(unrolled_matrix);
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