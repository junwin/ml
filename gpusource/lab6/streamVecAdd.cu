#include <iostream>
#include <wb.h>
#include <cuda_runtime.h>
using namespace std;
 


#define IS_WEBCUDA

#define NUM_STREAMS 4



#define HISTOGRAM_LENGTH 256

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            printf("Error:Failed to run stmt %s\n", #stmt);    \
            return -1;                                     \
        }                                                  \
    } while(0)




__global__ void vecAdd(float *in1, float *in2, float *out, int len) 
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < len)
	{
		out[i] = in1[i] + in2[i];
	}
}






int main(int argc, char ** argv) 
{
    wbArg_t args;
    int inputLength;
    float * hostInput1;
    float * hostInput2;
    float * hostOutput;
    float * deviceInput1;
    float * deviceInput2;
    float * deviceOutput;
    int segSize;

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput1 = (float *) wbImport(wbArg_getInputFile(args, 0), &inputLength);
    hostInput2 = (float *) wbImport(wbArg_getInputFile(args, 1), &inputLength);
    hostOutput = (float *) malloc(inputLength * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    cudaStream_t cudaStream[NUM_STREAMS];
    float * d_A[NUM_STREAMS];
    float * d_B[NUM_STREAMS];
    float * d_C[NUM_STREAMS];

    int segDataLen = inputLength/NUM_STREAMS;

    for(int i =0; i < NUM_STREAMS; i++)
    {
    	cudaStreamCreate(&cudaStream[i]);
    	wbCheck(cudaMalloc((void **)&d_A[i], segDataLen * sizeof(float)));
    	wbCheck(cudaMalloc((void **)&d_B[i], segDataLen * sizeof(float)));
    	wbCheck(cudaMalloc((void **)&d_C[i], segDataLen * sizeof(float)));
    	
    }


    int stepSize = segDataLen * NUM_STREAMS;
    int numIterations = inputLength / stepSize;
    int remainderSize = inputLength - numIterations * stepSize;

    printf("inpSize %d, stepSize %d, numIter %d, remdr %d \n", inputLength, stepSize, numIterations, remainderSize);

    for (int i =0; i < inputLength; i+= stepSize)
    {
    	for(int j =0; j< NUM_STREAMS; j++)
    	{
    		cudaMemcpyAsync(d_A[j], hostInput1 + (i + j*segDataLen), segDataLen * sizeof(float), cudaMemcpyHostToDevice, cudaStream[j]);
    		cudaMemcpyAsync(d_B[j], hostInput2 + (i + j*segDataLen), segDataLen * sizeof(float), cudaMemcpyHostToDevice, cudaStream[j]);		
    	} 

    	for(int j =0; j< NUM_STREAMS; j++)
    	{
    		vecAdd<<<segDataLen/256, 256,0, cudaStream[j]>>>(d_A[j], d_B[j], d_C[j], segDataLen);	
    	}

    	for(int j =0; j< NUM_STREAMS; j++)
    	{
    		cudaMemcpyAsync(hostOutput + (i + j*segDataLen), d_C[j], segDataLen * sizeof(float), cudaMemcpyDeviceToHost, cudaStream[j]);
    	}
    	
    }

    // handle the remainder
    int pos = numIterations * stepSize;
    if(remainderSize > 0)
    {
    	cudaMemcpyAsync(d_A[0], hostInput1 + pos, remainderSize * sizeof(float), cudaMemcpyHostToDevice, cudaStream[0]);
    	cudaMemcpyAsync(d_B[0], hostInput2 + pos, remainderSize * sizeof(float), cudaMemcpyHostToDevice, cudaStream[0]);
    	vecAdd<<<remainderSize/256, 256,0, cudaStream[0]>>>(d_A[0], d_B[0], d_C[0], remainderSize);	
    	cudaMemcpyAsync(hostOutput+ pos, d_C[0], remainderSize * sizeof(float), cudaMemcpyDeviceToHost, cudaStream[0]);	
    }


    wbSolution(args, hostOutput, inputLength);

    //free(hostInput1);
    //free(hostInput2);
    free(hostOutput);

    return 0;
}

