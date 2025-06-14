// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float *input, float *output, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float T[2*BLOCK_SIZE];
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int dx = blockDim.x;
  int i = 2 * (bx * dx + tx);
  int j = i + 1;
  int n = len;
  if(i < n) {
    T[2 * tx] = input[i];
  } else {
    T[2 * tx] = 0;
  }
  if(j < n) {
    T[2 * tx + 1] = input[j];
  } else {
    T[2 * tx + 1] = 0;
  }
  __syncthreads();

  int stride = 1;
  while(stride < 2*BLOCK_SIZE) {
    __syncthreads();
    int index = (threadIdx.x+1)*stride*2 - 1;
    if(index < 2*BLOCK_SIZE && (index-stride) >= 0)
      T[index] += T[index-stride];
    stride = stride*2;
  }

  stride = BLOCK_SIZE/2;
  while(stride > 0) {
    __syncthreads();
    int index = (threadIdx.x+1)*stride*2 - 1;
    if ((index+stride) < 2*BLOCK_SIZE)
      T[index+stride] += T[index];
    stride = stride / 2;
  }
  __syncthreads();
  if(i < n) {
    output[i] = T[2 * tx];
  }
  if(j < n) {
    output[j] = T[2 * tx + 1];
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  // Import data and create memory on host
  // The number of input elements in the input is numElements
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));


  // Allocate GPU memory.
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));


  // Clear output memory.
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));

  // Copy input memory to the GPU.
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));

  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid(ceil((float) numElements/(BLOCK_SIZE)), 1, 1);
  dim3 dimBlock(BLOCK_SIZE, 1, 1);

  
  for (int i = 0; i < dimGrid.x; i++) {
    float temp;
    if (i == 0) {
      scan<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, numElements);
    } else {
      cudaMemcpy(&temp, &deviceOutput[i * BLOCK_SIZE - 1], sizeof(float), cudaMemcpyDeviceToHost);
      temp += hostInput[i * BLOCK_SIZE];
      cudaMemcpy(&deviceInput[i * BLOCK_SIZE], &temp, sizeof(float), cudaMemcpyHostToDevice);
      scan<<<dimGrid, dimBlock>>>(&deviceInput[i * BLOCK_SIZE], &deviceOutput[i*BLOCK_SIZE], numElements);
    }
  }

  cudaDeviceSynchronize();

  // Copying output memory to the CPU
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));


  //@@  Free GPU Memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);


  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}

