#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include "matmul.h"

#define PERMUTE_BLOCK_SIZE 256
#define BLOCK_SIZE 16
#define TILE_WIDTH 16

__global__ void matrix_unrolling_kernel(const float *input, float *output,
                                        const int Batch, const int Channel,
                                        const int Height, const int Width,
                                        const int K) {
    /*
    Modify this function to implement the input matrix unrolling kernel.

    Function paramter definitions:
    input - input
    output - output
    Batch - batch_size (number of images in x)
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    // const int Width_unrolled = Height_out * Width_out;

    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    // #define out_4d(i2, i1, i0) output[(size_t)(i2) * (size_t)(Width_unrolled) + (size_t)(i1) * ((size_t)Batch *(size_t)(Width_unrolled)) + (size_t)i0]


    // TODO: Insert your input matrix unrolling kernel code here
    

    int h = (blockIdx.x / ((Width_out + TILE_WIDTH - 1) / TILE_WIDTH)) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.x % ((Width_out + TILE_WIDTH - 1) / TILE_WIDTH)) * TILE_WIDTH + threadIdx.x;
    int b = blockIdx.y;
    
    // int w_unroll = h * Width_out + w;

    if((h < Height_out) && (w < Width_out)){
            for(int c = 0; c < Channel; ++c){
                int w_base = c * (K*K);
                for(int p = 0; p < K; ++p) {
                    for(int q = 0; q < K; ++q) {
                        int h_unroll = w_base + p * K + q;
                        int w_unroll = b * Height_out * Width_out + h * Width_out + w;
                        if( h + p < Height && w + q < Width) {
                            output[w_unroll + h_unroll * Height_out * Width_out * Batch] = in_4d(b, c, h + p, w + q);
                        }
                        else {
                            output[w_unroll + h_unroll * Height_out * Width_out * Batch] = 0;
                        }
                    }
                }
            }
    }
    #undef in_4d
    // #undef out_4d
}


// Permutes the matmul result.
// The output feature map after matmul is of shape Map_out x Batch x Height_out x Width_out,
// and we need to permute it into Batch x Map_out x Height_out x Width_out.
// You don't need to modify this kernel.
__global__ void matrix_permute_kernel(const float *input, float *output, int Map_out,
                                      int Batch, int image_size) {
    int b = blockIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < image_size) {
        for (int m = 0; m < Map_out; m++) {
            output[b * Map_out * image_size + m * image_size + x] =
                    input[m * Batch * image_size + b * image_size + x];
        }
    }
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int Height_unrolled = Channel * K * K;
    const int Width_unrolled = Batch * Height_out * Width_out;
    float *unrolled_matrix;  // Pointer to device memory for storing the unrolled matrix
    float *matmul_output;

    cudaHostRegister((void *) host_input, sizeof(float) * Batch * Channel * Height * Width, cudaHostRegisterDefault);
    cudaHostRegister((void *) host_output, sizeof(float) * Batch * Map_out * (Height - K + 1) * (Width - K + 1), cudaHostRegisterDefault);


    cudaMalloc((void**)&unrolled_matrix, (size_t) Height_unrolled * Width_unrolled * sizeof(float));
    cudaMalloc((void**)&matmul_output, (Batch * Map_out * Height_out * Width_out) * sizeof(float));
    cudaMalloc((void **)device_input_ptr, Batch * Channel * Height * Width * sizeof(float));
    cudaMalloc((void **)device_mask_ptr, Map_out * Channel * K * K * sizeof(float));
    cudaMalloc((void **)device_output_ptr, Batch * Map_out * Height_out * Width_out * sizeof(float));

    // //cudaMemcpy(*device_input_ptr, host_input, Batch * Channel * Height * Width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, Map_out * Channel * K * K * sizeof(float), cudaMemcpyHostToDevice);

    //Stream creatation
    int steam_nums = 4;
    cudaStream_t streams[steam_nums];
    for (int i = 0; i < steam_nums; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Allocate device memory for each stream correspondign to the input size
    for (int i = 0; i < steam_nums; i++) {
        size_t offset = i * Batch * Channel * Height * Width / steam_nums;
        cudaMemcpyAsync(*device_input_ptr + offset, host_input + offset, Batch * Channel * Height * Width / steam_nums * sizeof(float), cudaMemcpyHostToDevice, streams[i]);
    }

    // Call the unrolling kernel for each stream
    for (int i = 0; i < steam_nums; i++) {
        size_t input_offset = (size_t)i * Batch * Channel * Height * Width / steam_nums;
        size_t unroll_offset = (size_t)i * Height_unrolled * Width_unrolled / steam_nums;
        dim3 block_unroll(TILE_WIDTH, TILE_WIDTH, 1);
        dim3 grid_unroll((Height_out + TILE_WIDTH - 1)/TILE_WIDTH * (Width_out + TILE_WIDTH - 1)/TILE_WIDTH, Batch / steam_nums, 1);
        matrix_unrolling_kernel<<<grid_unroll, block_unroll, 0, streams[i]>>>(*device_input_ptr + input_offset, unrolled_matrix + unroll_offset, Batch / steam_nums, Channel, Height, Width, K);
    }

    // Call the matrix multiplication kernel
    // Multiply the mask with the unrolled matrix
    for (int i = 0; i < steam_nums; i++) {

        int Height_unrolled = Channel * K * K;
        int Width_unrolled = Batch / steam_nums * Height_out * Width_out ;

        size_t unroll_offset = (size_t) i * Batch * Channel * K * K * Height_out * Width_out / steam_nums;
        size_t matmul_offset = (size_t) i * Batch * Map_out * Height_out * Width_out / steam_nums;

        dim3 matmul_grid_dim((Width_unrolled + TILE_WIDTH - 1)/ TILE_WIDTH, (Map_out + TILE_WIDTH - 1) / TILE_WIDTH, 1);
        dim3 matmul_block_dim(TILE_WIDTH, TILE_WIDTH, 1);
        matrixMultiplyShared<<<matmul_grid_dim, matmul_block_dim, 0, streams[i]>>>(*device_mask_ptr, unrolled_matrix + unroll_offset, matmul_output + matmul_offset, Map_out, Height_unrolled, Height_unrolled, Width_unrolled, Map_out, Width_unrolled);
    }
    

    // Call the permutation kernel for each stream
    for (int i = 0; i < steam_nums; i++) {
        size_t permute_offset = i * (Batch * Map_out * Height_out * Width_out / steam_nums);
        size_t image_size = Height_out * Width_out;
        dim3 permute_kernel_grid_dim((image_size - 1) / PERMUTE_BLOCK_SIZE + 1, Batch / steam_nums, 1);
        matrix_permute_kernel<<<permute_kernel_grid_dim, PERMUTE_BLOCK_SIZE, 0, streams[i]>>>(
            matmul_output + permute_offset, *device_output_ptr + permute_offset, Map_out, Batch / steam_nums, image_size
        );
    }

    // Copy the output back to host
    for (int i = 0; i < steam_nums; i++) {
        size_t offset_output = i * (Batch * Map_out * Height_out * Width_out / steam_nums);
        cudaMemcpyAsync((void*)(host_output + offset_output), *device_output_ptr + offset_output, Batch * Map_out * Height_out * Width_out / steam_nums * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }


    // Stream synchronization
    for (int i = 0; i < steam_nums; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    cudaFree(unrolled_matrix);
    cudaFree(matmul_output);
    cudaFree(*device_output_ptr);
    cudaFree(*device_input_ptr);
    cudaFree(*device_mask_ptr);
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // const int Height_out = Height - K + 1;
    // const int Width_out = Width - K + 1;
    // const int Height_unrolled = Channel * K * K;
    // const int Width_unrolled = Batch * Height_out * Width_out;

        // Pointer to device memory for storing the result of matrix multiplication
    
   
    

    // dim3 grid_unroll((Width_out + TILE_WIDTH - 1) / TILE_WIDTH, (Height_out + TILE_WIDTH - 1) / TILE_WIDTH, Batch);
    // dim3 block_unroll(TILE_WIDTH, TILE_WIDTH, 1);
    // matrix_unrolling_kernel<<<grid_unroll, block_unroll>>>(device_input, unrolled_matrix, Batch, Channel, Height, Width, K);
    
    // cudaDeviceSynchronize();

    // // Matrix multiplication and permutation. Do not modify.
    // // Multiply the mask with the unrolled matrix
    // dim3 matmul_grid_dim((Width_unrolled - 1) / MATMUL_TILE_WIDTH + 1,
    //                      (Map_out - 1) / MATMUL_TILE_WIDTH + 1, 1);
    // dim3 matmul_block_dim(MATMUL_TILE_WIDTH, MATMUL_TILE_WIDTH, 1);
    // matrixMultiplyShared<<<matmul_grid_dim, matmul_block_dim>>>(
    //     device_mask, unrolled_matrix, matmul_output, Map_out, Height_unrolled,
    //     Height_unrolled, Width_unrolled, Map_out, Width_unrolled
    // );

    // // Permute the result of matrix multiplication
    // const int out_image_size = Height_out * Width_out;
    // dim3 permute_kernel_grid_dim((out_image_size - 1) / PERMUTE_BLOCK_SIZE + 1, Batch, 1);
    // matrix_permute_kernel<<<permute_kernel_grid_dim, PERMUTE_BLOCK_SIZE>>>(
    //     matmul_output, device_output, Map_out, Batch, out_image_size
    // );

    // cudaFree(matmul_output);
    // cudaFree(unrolled_matrix);
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Copy the output back to host

    // TODO: Free device memory
    // const int Height_out = Height - K + 1;
    // const int Width_out = Width - K + 1;
    // cudaMemcpy(host_output, device_output, Batch * Map_out * Height_out * Width_out * sizeof(float), cudaMemcpyDeviceToHost);
    // cudaFree(device_output);
    // cudaFree(device_input);
    // cudaFree(device_mask);

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