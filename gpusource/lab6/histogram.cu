#include <iostream>
#include <wb.h>
#include <cuda_runtime.h>
using namespace std;
 

#define IS_DEBUG
#define IS_WEBCUDA
#define HISTOGRAM_LENGTH 256

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            printf("Error:Failed to run stmt %s\n", #stmt);    \
            return -1;                                     \
        }                                                  \
    } while(0)

__global__ void float2uchar(float *input, unsigned char * output, int size) 
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if(i < size)
	{
		output[i] = (unsigned char) (255 * input[i]);
	}
}

__global__ void uchar2Float(unsigned char *input, float * output, int size) 
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if(i < size)
	{
		output[i] = ((float) input[i])/255;
	}
}


__global__ void ucharCorrect2Float(unsigned char *input, float * output, unsigned int * correction, int size) 
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if(i < size)
	{
		unsigned char a = correction[input[i]];
		output[i] = ((float) a)/255;
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

void HostHistogram(unsigned char *buffer, long size, unsigned int *histo)
{
	for(long i =0; i<HISTOGRAM_LENGTH; i++)
	{
		histo[i]=0;
	}
	for(long i =0; i<size; i++)
	{
		unsigned int bufferIndex = (unsigned int ) buffer[i];
		histo[bufferIndex] += 1;
	}
}

__global__ void histo(unsigned char *buffer, long size, unsigned int *histo)
{
	__shared__ unsigned int histo_private[HISTOGRAM_LENGTH];
	
	if(threadIdx.x < HISTOGRAM_LENGTH)
		histo_private[threadIdx.x] = 0;

	__syncthreads();
		
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	
	// stride is total number of threads
	int stride = blockDim.x *gridDim.x;
	
	while ( i < size)
	{
		unsigned int bIdx = (unsigned int)buffer[i];
		atomicAdd(&(histo_private[bIdx]), 1);
		//atomicAdd( &(histo[bIdx]), 1);
		i += stride;
	}
	
	// wait for all other threads in the block to finish
	__syncthreads();
	
	if(threadIdx.x < HISTOGRAM_LENGTH)
	{
		atomicAdd( &(histo[threadIdx.x]), histo_private[threadIdx.x]);
	}
}


__global__ void applyEdgeContibutions(float *input, float *aux, int len) 
{
    unsigned int t = threadIdx.x;
    unsigned int start = 2 * blockIdx.x * BLOCK_SIZE;

    if (blockIdx.x) 
    {
       if (start + t < len)
          input[start + t] += aux[blockIdx.x - 1];

       if (start + BLOCK_SIZE + t < len)
          input[start + BLOCK_SIZE + t] += aux[blockIdx.x - 1];
    }
}

__global__ void scan(float * input, float * output, float *aux, int len) 
{
    // Load a segment of the input vector into shared memory
    __shared__ float scan_array[BLOCK_SIZE << 1];

    unsigned int t = threadIdx.x;
    unsigned int start = 2 * blockIdx.x * BLOCK_SIZE;

    if (start + t < len)
       scan_array[t] = input[start + t];
    else
       scan_array[t] = 0;

    if (start + BLOCK_SIZE + t < len)
       scan_array[BLOCK_SIZE + t] = input[start + BLOCK_SIZE + t];
    else
       scan_array[BLOCK_SIZE + t] = 0;

    __syncthreads();

    // Reduction
    int stride;
    for (stride = 1; stride <= BLOCK_SIZE; stride <<= 1) 
    {
       int index = (t + 1) * stride * 2 - 1;

       if (index < 2 * BLOCK_SIZE)
          scan_array[index] += scan_array[index - stride];

       __syncthreads();
    }

    // Post reduction
    for (stride = BLOCK_SIZE >> 1; stride; stride >>= 1) 
    {
       int index = (t + 1) * stride * 2 - 1;

       if (index + stride < 2 * BLOCK_SIZE)
          scan_array[index + stride] += scan_array[index];

       __syncthreads();
    }

    if (start + t < len)
       output[start + t] = scan_array[t];

    if (start + BLOCK_SIZE + t < len)
       output[start + BLOCK_SIZE + t] = scan_array[BLOCK_SIZE + t];

    if (aux && t == 0)
       aux[blockIdx.x] = scan_array[2 * BLOCK_SIZE - 1];
}


void Scan(float * hostInput, float * hostOutput, int numElements)
{
    float * deviceInput;
    float * deviceOutput;
    float *deviceEdgeValsArray;
    float *deviceEdgeValsScanArray;

    cudaHostAlloc(&hostOutput, numElements * sizeof(float), cudaHostAllocDefault);
  
    cudaMalloc((void**)&deviceInput, numElements*sizeof(float));
    cudaMalloc((void**)&deviceOutput, numElements*sizeof(float));

    cudaMalloc((void**)&deviceEdgeValsArray, (BLOCK_SIZE << 1) * sizeof(float));
    cudaMalloc((void**)&deviceEdgeValsScanArray, (BLOCK_SIZE << 1) * sizeof(float));

    cudaMemset(deviceOutput, 0, numElements*sizeof(float));

    cudaMemcpy(deviceInput, hostInput, numElements*sizeof(float), cudaMemcpyHostToDevice);

    int numBlocks = ceil((float)numElements/(BLOCK_SIZE<<1));

    dim3 dimGrid(numBlocks, 1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);

    //@@ Modify this to complete the functionality of the scan
    //@@ on the deivce

    scan<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, deviceEdgeValsArray, numElements);
    cudaDeviceSynchronize();

    scan<<<dim3(1,1,1), dimBlock>>>(deviceEdgeValsArray, deviceEdgeValsScanArray, NULL, BLOCK_SIZE << 1);
    cudaDeviceSynchronize();

    applyEdgeContibutions<<<dimGrid, dimBlock>>>(deviceOutput, deviceEdgeValsScanArray, numElements);
    cudaDeviceSynchronize();

    cudaMemcpy(hostOutput, deviceOutput, numElements*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    cudaFree(deviceEdgeValsArray);
    cudaFree(deviceEdgeValsScanArray);

 
}





void HostCDF(unsigned int * histo, float * cdf, long numHiostoElems, long numPixels)
{
	cdf[0] = histo[0]/numPixels;
	for(int i =1; i< numHiostoElems; i++)
	{
		cdf[i] = cdf[i-1] + ((float)histo[i])/numPixels;

	}
}

float HostGetMin(float * input, long numElems)
{
	unsigned int minVal = input[0];
	for(long i = 0; i < numElems; i++)
	{
		if(input[i] < minVal)
			minVal = input[i];
	}

	return minVal;
}

unsigned int Clamp(unsigned int x, unsigned int start, unsigned int end)
{
	unsigned int temp =start;
	if (x > start)
		temp = x;

	if(temp > end)
		temp = end;

	return temp;
}

unsigned int CorrectedColorLevel(unsigned int value, float * cdf, float cdfMin)
{
	return Clamp(255*(cdf[value] - cdfMin)/(1-cdfMin), 0, 255);
}



void HostUcharCorrect2Float(unsigned char *input, float * output, unsigned int * correction, int size) 
{
	for(long i = 0; i < size; i++)
	{
		unsigned char a = correction[input[i]];
		output[i] = ((float) a)/255;
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

    wbCheck(cudaMalloc((void **)&d_FA, dataSize * sizeof(float)));	
    wbCheck(cudaMalloc((void **)&d_UA, dataSize * sizeof(unsigned char)));	
    wbCheck(cudaMalloc((void **)&d_GI, imageHeight * imageWidth * sizeof(unsigned char)));
	wbCheck(cudaMalloc((void **)&d_HD, HISTOGRAM_LENGTH * sizeof(unsigned int)));

	wbCheck(cudaMemcpy(d_FA, hostInputImageData, dataSize*sizeof(float), cudaMemcpyHostToDevice));

	// Launch the CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(dataSize + threadsPerBlock - 1) / threadsPerBlock;

    printf("CUDA float2uchar launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    float2uchar<<<blocksPerGrid, threadsPerBlock>>>(d_FA, d_UA, dataSize);
    err = cudaGetLastError();

 	if (err != cudaSuccess)
	{
    	fprintf(stderr, "Failed convert float to uchar (error code %s)!\n", cudaGetErrorString(err));
    	exit(EXIT_FAILURE);
	}

	wbCheck(cudaMemcpy(ucharArray, d_UA, dataSize * sizeof(unsigned char), cudaMemcpyDeviceToHost));

	

	// Diagnostics
	#if defined(IS_DEBUG)

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
	#endif


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

	wbCheck(cudaMemcpy(grayImage, d_GI, imageHeight * imageWidth * sizeof(unsigned char), cudaMemcpyDeviceToHost));

	

		// Diagnostics
	#if defined(IS_DEBUG)

	for(int i =0; i < 24; i++)
	{
		printf("%d : %u : %u\n", i, ucharArray[i], grayImage[i]);	
	}

	#endif



	// Launch the CUDA Kernel - histo
	
	threadsPerBlock = 512;
	blocksPerGrid =(imageHeight * imageWidth + threadsPerBlock - 1) / threadsPerBlock;

	printf("CUDA kernel histo launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

	histo<<<blocksPerGrid, threadsPerBlock>>>(d_GI, imageHeight * imageWidth, d_HD);
	err = cudaGetLastError();

 	if (err != cudaSuccess)
	{
    	fprintf(stderr, "Failed histo (error code %s)!\n", cudaGetErrorString(err));
    	exit(EXIT_FAILURE);
	}

	wbCheck(cudaMemcpy(histoBins, d_HD, HISTOGRAM_LENGTH * sizeof(unsigned int), cudaMemcpyDeviceToHost));


	// Diagnostics
#if defined(IS_DEBUG)

	for(int i =0; i < HISTOGRAM_LENGTH; i++)
	{
		printf("%d %u\n", i, histoBins[i]);	
	}

	unsigned int * testHistoBins =  (unsigned int *)malloc(HISTOGRAM_LENGTH * sizeof(unsigned int));
	printf("call host calc\n");
	HostHistogram(grayImage, imageHeight * imageWidth, testHistoBins);
	printf("compare kernel and host calc\n");
	long sumHost = 0;
	long sumGpu = 0;
	for(int i =0; i < HISTOGRAM_LENGTH; i++)
	{
		sumHost += testHistoBins[i];
		sumGpu += histoBins[i];
		if(histoBins[i] != testHistoBins[i])
		{
			printf("Error: index: %d GPUBins:  %u TestBins: %u\n", i, histoBins[i], testHistoBins[i]);
		}
			
	}

	printf("Height:%d Width%d Sum host = %ld SumGpu = %ld\n", imageHeight, imageWidth, sumHost, sumGpu);

#endif


	printf("Get CDF\n");
	float * cdf =   (float *)malloc(HISTOGRAM_LENGTH * sizeof(float));

	HostCDF(histoBins,  cdf, HISTOGRAM_LENGTH, imageHeight * imageWidth);
	float cdfMin = HostGetMin(cdf, HISTOGRAM_LENGTH);

#if defined(IS_DEBUG)

	for(int i =0; i < HISTOGRAM_LENGTH; i++)
	{
		printf("cdf: %d:%f\n", i, cdf[i]);	
	}
	printf("min cdf: %f\n", cdfMin);	


#endif

	//  should be a kernel to create array of corrections
	printf("get host correction array\n");
	unsigned int * correction = (unsigned int *) malloc(HISTOGRAM_LENGTH * sizeof(unsigned int));
	for(int i = 0; i < HISTOGRAM_LENGTH; i++)
	{
		correction[i] = CorrectedColorLevel((unsigned int) i, cdf, cdfMin);
	}


#if defined(IS_DEBUG)

	for(int i =0; i < HISTOGRAM_LENGTH; i++)
	{
		printf("correction: %d:%u\n", i, correction[i]);	
	}
	


#endif



	printf("Attempt correction\n");
	hostOutputImageData = wbImage_getData(outputImage);
	HostUcharCorrect2Float(ucharArray, hostOutputImageData, correction, dataSize); 
	//void ucharCorrect2Float(unsigned char *input, hostOutputImageData, unsigned int * correction, int size) 

	// Diagnostics
#if defined(IS_DEBUG)
	printf("Compare with expected output image\n");
	wbImage_t testImage;
	float * hostTestImageData;

	const char * testImageFile = "output.ppm";


	testImage = wbImport(testImageFile);
	hostTestImageData = wbImage_getData(testImage); 
	hostInputImageData = wbImage_getData(inputImage);
	
	long errorCount =0;
	for(long i =0; i < dataSize; i++)
	{
		if(hostTestImageData[i] != hostOutputImageData[i])
		{
			errorCount++;
			if(errorCount < 30)
			{
				printf("i: %ld input:%f uchar:%u expected:%f actual:%f\n", i, hostInputImageData[i], ucharArray[i], hostTestImageData[i],hostOutputImageData[i]);	
			}
		}
	}

	if(errorCount >0)
	{
		printf("output and test image do not match %ld \n", errorCount);
	}

#endif

	err = cudaFree(d_FA);
	err = cudaFree(d_UA);
	err = cudaFree(d_GI);
	err = cudaFree(d_HD);
	free(ucharArray);
	free(grayImage);
	free(histoBins);

}
