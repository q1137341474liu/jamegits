#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define TILE_WIDTH 4
#define MASK_WIDTH 3
//@@ Define constant memory for device kernel here
__constant__ float deviceKernel[MASK_WIDTH*MASK_WIDTH*MASK_WIDTH];
__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here

  int x = blockIdx.x * TILE_WIDTH + threadIdx.x - 1;
  int y = blockIdx.y * TILE_WIDTH + threadIdx.y - 1;
  int z = blockIdx.z * TILE_WIDTH + threadIdx.z - 1;

  // Shared memory for input
  __shared__ float SM[TILE_WIDTH + 2][TILE_WIDTH + 2][TILE_WIDTH + 2];

  // Read input elements into shared memory
  if ((x >= 0 && x < x_size) && (y >= 0 && y < y_size) && (z >= 0 && z < z_size)) {
    SM[threadIdx.z][threadIdx.y][threadIdx.x] = input[x + x_size * y + x_size * y_size * z];
  } else {
    SM[threadIdx.z][threadIdx.y][threadIdx.x] = 0;
  }
  
  // make sure all threads have read the input
  __syncthreads();


  if (threadIdx.x >= 1 && threadIdx.x < TILE_WIDTH + 1 && threadIdx.y >= 1 && threadIdx.y < TILE_WIDTH + 1 && threadIdx.z >= 1 && threadIdx.z < TILE_WIDTH + 1 && x < x_size && y < y_size && z < z_size) {
    float result = 0;

    for (int i = -1; i <= 1; i++) {
      for (int j = -1; j <= 1; j++) {
        for (int k = -1; k <= 1; k++) {
          result = result + SM[threadIdx.z + i][threadIdx.y + j][threadIdx.x + k] * deviceKernel[(k + 1) + 3 * (j + 1) + 9 * (i + 1)];
        }
      }
    output[z * y_size * x_size + y * x_size + x] = result;
    }
  }
 }

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  //@@ Initial deviceInput and deviceOutput here.

  float *deviceInput;
  float *deviceOutput;
  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);


  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  cudaMalloc((void**) &deviceInput, (inputLength-3) * sizeof(float));
  cudaMalloc((void**) &deviceOutput, (inputLength-3) * sizeof(float));

  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpy(deviceInput, hostInput + 3, (inputLength-3) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(deviceKernel, hostKernel, kernelLength*sizeof(float), 0, cudaMemcpyHostToDevice);


  //@@ Initialize grid and block dimensions here
  dim3 dimGrid(ceil((float)(x_size) / (TILE_WIDTH)), ceil((float)(y_size) / (TILE_WIDTH)), ceil((float)(z_size) / (TILE_WIDTH)));
  dim3 dimBlock(TILE_WIDTH + MASK_WIDTH - 1, TILE_WIDTH + MASK_WIDTH - 1, TILE_WIDTH + MASK_WIDTH - 1);

  //@@ Launch the GPU kernel here
  conv3d<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
  cudaDeviceSynchronize();



  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(hostOutput+3, deviceOutput, (inputLength-3)*sizeof(float), cudaMemcpyDeviceToHost);

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  //@@ Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}