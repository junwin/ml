#include <iostream>
#include <wb.h>
#include <cuda_runtime.h>
using namespace std;
 


#define HISTOGRAM_LENGTH 256

__global__ void float2uchar(float *input, unsigned char * output, int size) 
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if(i < size)
	{
		output[i] = (unsigned char) (255 * input[i]);
	}
}

__global__ void rgb2gray(unsigned char * ucharImage, unsigned char * grayImage, int size) 
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int rgbIdx = i*3;

	if(i < size)
	{
		uint r = ucharImage[rgbIdx];
		uint g = ucharImage[rgbIdx+1];
		uint b = ucharImage[rgbIdx+2];
		grayImage[i]= (unsigned char) (0.21*r + 0.71*g + 0.07*b);
	}
}


__global__ void histo(unsigned char *buffer, long size, unsigned int *histo)
{
	__shared__ unsigned int histo_private[HISTOGRAM_LENGTH];
	
	if(threadIdx.x < 256)
		histo_private[threadIdx.x] = 0;
		
	int i = threadIdx.x * blockIdx.x * blockDim.x;
	
	// stride is total number of threads
	int stride = blockDim.x *gridDim.x;
	
	while ( i < size)
	{
		atomicAdd(&(histo_private[buffer[i]]), 1);
		i += stride;
	}
	
	// wait for all other threads in the block to finish
	__syncthreads();
	
	if(threadIdx.x < HISTOGRAM_LENGTH)
	{
		atomicAdd( &(histo[threadIdx.x]), histo_private[threadIdx.x]);
	}
}

int main(int argc, char ** argv) 
{
    //wbArg_t args;
    int imageWidth;
    int imageHeight;
    int imageChannels;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    const char * inputImageFile;


 //@@ Insert more code here

    //args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = "input.ppm";


	inputImage = wbImport(inputImageFile);
	imageWidth = wbImage_getWidth(inputImage);
	imageHeight = wbImage_getHeight(inputImage);
	imageChannels = wbImage_getChannels(inputImage);
	hostInputImageData = wbImage_getData(inputImage);
	outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

	int dataSize = imageHeight * imageWidth * imageChannels;

	printf("Input %s width %d height %d channels %d dataSize %d\n",inputImageFile, imageWidth, imageHeight, imageChannels, dataSize);

	
	cudaError_t err = cudaSuccess;

	
	unsigned char * ucharArray = (unsigned char *)malloc(dataSize*sizeof(unsigned char));
	unsigned char * grayImage = (unsigned char *)malloc(imageHeight * imageWidth * sizeof(unsigned char));
	unsigned int * histoBins =  (unsigned int *)malloc(HISTOGRAM_LENGTH * sizeof(unsigned int));

	float *d_FA = NULL;
	unsigned char *d_UA = NULL;
	unsigned char *d_GI = NULL;
	unsigned int * d_HD = NULL;

    	err = cudaMalloc((void **)&d_FA, dataSize * sizeof(float));	
    	err = cudaMalloc((void **)&d_UA, dataSize * sizeof(unsigned char));	
    	err = cudaMalloc((void **)&d_GI, imageHeight * imageWidth * sizeof(unsigned char));
	err = cudaMalloc((void **)&d_HD, HISTOGRAM_LENGTH * sizeof(unsigned int));

	err = cudaMemcpy(d_FA, hostInputImageData, dataSize*sizeof(float), cudaMemcpyHostToDevice);

	// Launch the CUDA Kernel
    	int threadsPerBlock = 256;
    	int blocksPerGrid =(dataSize + threadsPerBlock - 1) / threadsPerBlock;

    	printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    	float2uchar<<<blocksPerGrid, threadsPerBlock>>>(d_FA, d_UA, dataSize);
    	err = cudaGetLastError();

 	if (err != cudaSuccess)
    	{
        	fprintf(stderr, "Failed convert float to uchar (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}

	err = cudaMemcpy(ucharArray, d_UA, dataSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	if (err != cudaSuccess)
    	{
        	fprintf(stderr, "Failed to copy uchar data from device to host (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}

	// Diagnostics
	for(int i =0; i < 10; i++)
	{
		printf("%f : %u : %u\n", hostInputImageData[i], ucharArray[i], (unsigned char) (255 * hostInputImageData[i]));	
	}

	for(int i =0; i < dataSize; i++)
	{
		if((unsigned char) (255 * hostInputImageData[i]) != ucharArray[i])
		{
			printf("Error pos %d : %f : %u : %u\n", i, hostInputImageData[i], ucharArray[i], (unsigned char) (255 * hostInputImageData[i]));	
		}
	}


	// Launch the CUDA Kernel - gray scale
    	threadsPerBlock = 256;
    	blocksPerGrid =(dataSize + threadsPerBlock - 1) / threadsPerBlock;

    	printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    	rgb2gray<<<blocksPerGrid, threadsPerBlock>>>(d_UA, d_GI, dataSize);
    	err = cudaGetLastError();

 	if (err != cudaSuccess)
    	{
        	fprintf(stderr, "Failed greyImage (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}

	err = cudaMemcpy(grayImage, d_GI, imageHeight * imageWidth * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	if (err != cudaSuccess)
    	{
        	fprintf(stderr, "Failed to copy from device to host (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}

	// Diagnostics
	for(int i =0; i < 24; i++)
	{
		printf("%u : %u\n", ucharArray[i], grayImage[i]);	
	}



	// Launch the CUDA Kernel - histo
    	threadsPerBlock = 256;
    	blocksPerGrid =(imageHeight * imageWidth + threadsPerBlock - 1) / threadsPerBlock;

    	printf("CUDA kernel histo launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    	histo<<<blocksPerGrid, threadsPerBlock>>>(d_GI, imageHeight * imageWidth, d_HD);
    	err = cudaGetLastError();

 	if (err != cudaSuccess)
    	{
        	fprintf(stderr, "Failed histo (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}

	err = cudaMemcpy(histoBins, d_HD, HISTOGRAM_LENGTH, cudaMemcpyDeviceToHost);

	if (err != cudaSuccess)
    	{
        	fprintf(stderr, "Failed to copy histogram data from device to host (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}

	// Diagnostics
	for(int i =0; i < HISTOGRAM_LENGTH; i++)
	{
		printf("%d\n", histoBins[i]);	
	}



	err = cudaFree(d_FA);
	err = cudaFree(d_UA);
	err = cudaFree(d_GI);
	err = cudaFree(d_HD);
	free(ucharArray);
	free(grayImage);
	free(histoBins);

}
