#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define TILE_WIDTH 16

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  __shared__ float subTileA[TILE_WIDTH][TILE_WIDTH];
  __shared__ float subTileB[TILE_WIDTH][TILE_WIDTH];
  int bx = blockIdx.x; 
  int by = blockIdx.y;
  int tx = threadIdx.x; 
  int ty = threadIdx.y;
// Identify the row and column of the P element to work on
  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;
  float Pvalue = 0;
  //@@ You have to use shared memory for this MP
  for (int q = 0; q < (numAColumns + TILE_WIDTH - 1) / TILE_WIDTH; ++q) {
    if (Row < numARows && (q * TILE_WIDTH + tx) < numAColumns)
        subTileA[ty][tx] = A[Row*numAColumns + q*TILE_WIDTH+tx];
    else
        subTileA[ty][tx] = 0;
    if (Col < numBColumns && (q * TILE_WIDTH + ty) < numBRows)
        subTileB[ty][tx] = B[(q*TILE_WIDTH+ty)*numBColumns+Col];
    else
        subTileB[ty][tx] = 0;
  // subTileA[ty][tx] = A[Row*numAColumns + q*TILE_WIDTH+tx];
  // subTileB[ty][tx] = B[(q*TILE_WIDTH+ty)*numBColumns+Col];
  __syncthreads();
  for (int k = 0; k < TILE_WIDTH; ++k)
    Pvalue += subTileA[ty][k] * subTileB[k][tx];
  __syncthreads();
  }
  if (Row < numCRows && Col < numCColumns)
    C[Row*numCColumns+Col] = Pvalue;
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix

  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  //@@ Importing data and creating memory on host
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;

  //@@ Allocate the hostC matrix
  hostC = (float *)malloc(numCRows * numCColumns * sizeof(float));  

  //@@ Allocate GPU memory here
  int sizeA = numARows * numAColumns * sizeof(float);
  int sizeB = numBRows * numBColumns * sizeof(float);
  int sizeC = numCRows * numCColumns * sizeof(float);
  float *A_d, *B_d, *C_d;

  cudaMalloc((void **) &A_d, sizeA);
  cudaMemcpy(A_d, hostA, sizeA, cudaMemcpyHostToDevice);
  cudaMalloc((void **) &B_d, sizeB);
  cudaMemcpy(B_d, hostB, sizeB, cudaMemcpyHostToDevice);
  cudaMalloc((void **) &C_d, sizeC);

  //@@ Copy memory to the GPU here


  //@@ Initialize the grid and block dimensions here
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 dimGrid((numCColumns + TILE_WIDTH - 1) / TILE_WIDTH,(numCRows + TILE_WIDTH - 1) / TILE_WIDTH, 1);

  //@@ Launch the GPU Kernel here
  matrixMultiplyShared <<<dimGrid, dimBlock>>> (A_d, B_d, C_d, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
  cudaDeviceSynchronize();

  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, C_d, sizeC, cudaMemcpyDeviceToHost);

  //@@ Free the GPU memory here
  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);

  //@@ Free the hostC matrix
  free(hostC);
  return 0;
}
