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
		//output[i] = (unsigned char) (i%255);
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

	
	unsigned char * charArray = (unsigned char *)malloc(dataSize*sizeof(unsigned char));

	float *d_FA = NULL;
    	err = cudaMalloc((void **)&d_FA, dataSize*sizeof(float));

	unsigned char *d_UA = NULL;
    	err = cudaMalloc((void **)&d_UA, dataSize*sizeof(unsigned char));

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

	err = cudaMemcpy(charArray, d_UA, dataSize*sizeof(unsigned char), cudaMemcpyDeviceToHost);

	if (err != cudaSuccess)
    	{
        	fprintf(stderr, "Failed to copy from device to host (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}

	// Diagnostics
	for(int i =0; i < 10; i++)
	{
		printf("%f : %u : %u\n", hostInputImageData[i], charArray[i], (unsigned char) (255 * hostInputImageData[i]));	
	}

	for(int i =0; i < dataSize; i++)
	{
		if((unsigned char) (255 * hostInputImageData[i]) != charArray[i])
		{
			printf("Error pos %d : %f : %u : %u\n", i, hostInputImageData[i], charArray[i], (unsigned char) (255 * hostInputImageData[i]));	
		}
	}


	err = cudaFree(d_FA);
	err = cudaFree(d_UA);
	free(charArray);

}
