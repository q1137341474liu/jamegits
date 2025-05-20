// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256

//@@ insert code here
__global__ void float_to_unsign( float *inputImage, unsigned char *outputImage, int imageWidth, int imageHeight, int imageChannels) {
  
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int tz = threadIdx.z;
  int index = imageChannels * (y * imageWidth + x) + tz;
  if (x < imageWidth && y < imageHeight) {
    
    outputImage[index] = (unsigned char)(inputImage[index] * 255);
  }
}

__global__ void rgb_to_gray( unsigned char *ucharImage, unsigned char *grayImage, int imageWidth, int imageHeight, int imageChannels) {
  
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index = y * imageWidth + x;
  if (x < imageWidth && y < imageHeight) {
	
		unsigned char r = ucharImage[imageChannels*index];
		unsigned char g = ucharImage[imageChannels*index + 1];
		unsigned char b = ucharImage[imageChannels*index + 2];
		grayImage[index] = (unsigned char) (0.21*r + 0.71*g + 0.07*b);
  }
}

__global__ void hist_of_gray( unsigned char *grayImage, int *histogram, int imageWidth, int imageHeight) {
  
  __shared__ unsigned int histo[256];
  int index = blockDim.x * threadIdx.y + threadIdx.x;
  if (index < 256) {
    histo[index] = 0;
  }
  __syncthreads();
  

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < imageWidth && y < imageHeight) {
    int pixel_index  = y * imageWidth + x;
    atomicAdd(&histo[grayImage[pixel_index]], 1);
  }
  __syncthreads();

  if (index < 256) {
    atomicAdd(&histogram[index], histo[index]);
  }
}

__global__ void cdf_of_hist(int *histogram, float *cdf, int imageWidth, int imageHeight) {
  __shared__ float histo[HISTOGRAM_LENGTH];
  int tx = threadIdx.x;
  int bx = blockIdx.x;

  histo[2*tx] = (float) histogram[2*(bx*blockDim.x + tx)] / (1.0f * imageWidth * imageHeight);
  histo[2*tx + 1] = (float) histogram[2*(bx*blockDim.x + tx) + 1] / (1.0f * imageWidth * imageHeight);
  __syncthreads();

  int stride = 1;
  while (stride < 256){
    __syncthreads();
    int index = (tx+1)*stride*2-1;
    if(index < 256 && index >= stride){
      histo[index] += histo[index - stride];
    }
    stride*=2;
  }

  stride = 64;
  while (stride >0){
    __syncthreads();
    int index = (tx+1) * stride * 2-1;
    if((index+stride) < 256){
      histo[index + stride] +=histo[index];
    }
    stride/=2;
  }
  __syncthreads();

  cdf[2*(bx*blockDim.x + tx)]=histo[tx*2];
  cdf[2*(bx*blockDim.x + tx) + 1]=histo[tx*2+1];
}

__global__ void equalize( unsigned char *grayImage, float *cdf, unsigned char *outputImage, int imageWidth, int imageHeight, int imageChannels) {
  
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int tz = threadIdx.z;
  int index = imageChannels * (y * imageWidth + x) + tz;

  if (x < imageWidth && y < imageHeight){
    grayImage[index] = min(max(255 * (cdf[grayImage[index]] - cdf[0])/(1.0 - cdf[0]), 0.0), 255.0);
  }
}

__global__ void back_to_float( unsigned char *inputImage, float *outputImage, int imageWidth, int imageHeight, int imageChannels) {
  
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int tz = threadIdx.z;
  int index = imageChannels * (y * imageWidth + x) + tz;

  if (x < imageWidth && y < imageHeight) {
    outputImage[index] = (float)(inputImage[index]) / 255.0;
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float *deviceInput;
  float *deviceOutput;
  float *cdf;
  unsigned char *ucharImage; //three channels
  unsigned char *grayImage; //one channel
  int *histogram;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  //Import data and create memory on host
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);


  //@@ insert code here
  cudaMalloc((void **)&deviceInput, imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void **)&deviceOutput, imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void **)&ucharImage, imageWidth * imageHeight * imageChannels * sizeof(unsigned char));
  cudaMalloc((void **)&grayImage, imageWidth * imageHeight * sizeof(unsigned char));
  cudaMalloc((void **)&histogram, HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMalloc((void **)&cdf, HISTOGRAM_LENGTH * sizeof(float));
  cudaMemset(histogram, 0, HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMemset(cdf, 0, HISTOGRAM_LENGTH * sizeof(float));
  cudaMemcpy(deviceInput, hostInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice);

  

  //@@ insert code here
  dim3 dimGrid(ceil( (float)imageWidth / (1.0 * 16)), ceil( (float)imageHeight / (1.0 * 16)), 1);
  dim3 dimBlock(16, 16, imageChannels);
  float_to_unsign<<<dimGrid, dimBlock>>>(deviceInput, ucharImage, imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();

  dim3 dimGrid2(ceil( (float)imageWidth / (1.0 * 16)), ceil( (float)imageHeight / (1.0 * 16)), 1);
  dim3 dimBlock2(16, 16, 1);
  rgb_to_gray<<<dimGrid2, dimBlock2>>>(ucharImage, grayImage, imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();

  dim3 dimGrid3(ceil( (float)imageWidth / (1.0 * 16)), ceil( (float)imageHeight / (1.0 * 16)), 1);
  dim3 dimBlock3(16, 16, 1);
  hist_of_gray<<<dimGrid3, dimBlock3>>>(grayImage, histogram, imageWidth, imageHeight);
  cudaDeviceSynchronize();

  dim3 dimGrid4(1, 1, 1);
  dim3 dimBlock4(128, 1, 1);
  cdf_of_hist<<<dimGrid4, dimBlock4>>>(histogram, cdf, imageWidth, imageHeight);
  cudaDeviceSynchronize();

  dim3 dimGrid5(ceil( (float)imageWidth / (1.0 * 16)), ceil( (float)imageHeight / (1.0 * 16)), 1);
  dim3 dimBlock5(16, 16, imageChannels);
  equalize<<<dimGrid5, dimBlock5>>>(ucharImage, cdf, ucharImage, imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();

  dim3 dimGrid6(ceil( (float)imageWidth / (1.0 * 16)), ceil( (float)imageHeight / (1.0 * 16)), 1);
  dim3 dimBlock6(16, 16, imageChannels);
  back_to_float<<<dimGrid6, dimBlock6>>>(ucharImage, deviceOutput, imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();

  cudaMemcpy(hostOutputImageData, deviceOutput, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);

  
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(ucharImage);
  cudaFree(grayImage);
  cudaFree(histogram);
  cudaFree(cdf);
  wbSolution(args, outputImage);

  free(hostInputImageData);
  free(hostOutputImageData);


  return 0;
}

