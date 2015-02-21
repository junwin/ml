
#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            return -1;                                     \
        }                                                  \
    } while(0)

__global__ void sumReduction(float * input, float * output, int len) 
{
    __shared__ float pSum[2 * BLOCK_SIZE];
    unsigned int i = threadIdx.x, start = 2 * blockIdx.x * BLOCK_SIZE;

    if (start + i < len)
       pSum[i] = input[start + i];
    else
       pSum[i] = 0;

    if (start + BLOCK_SIZE + i < len)
       pSum[BLOCK_SIZE + i] = input[start + BLOCK_SIZE + i];
    else
       pSum[BLOCK_SIZE + i] = 0;

    for (unsigned int stride = BLOCK_SIZE; stride >= 1; stride >>= 1) 
    {
       __syncthreads();
       if (i < stride)
          pSum[i] += pSum[i+stride];
    }

    if (i == 0)
       output[blockIdx.x] = pSum[0];
}

int main(int argc, char ** argv) 
{
    int ii;
    wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * deviceInput;
    float * deviceOutput;
    int numInputElements; // number of elements in the input list
    int numOutputElements; // number of elements in the output list

    numInputElements = 9999;
    hostInput = (float*) malloc(numInputElements * sizeof(float));
    float answer = 0.0;
    for(int i = 0; i < numInputElements; i++)
    {
        hostInput[i] = i;
        answer += hostInput[i];
    }

    printf("Expected answer %f\n", answer);

    // args = wbArg_read(argc, argv);


    //hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numInputElements);

    numOutputElements = numInputElements / (BLOCK_SIZE<<1);
    if (numInputElements % (BLOCK_SIZE<<1)) 
    {
        numOutputElements++;
    }

    hostOutput = (float*) malloc(numOutputElements * sizeof(float));



    //wbLog(TRACE, "The number of input elements in the input is ", numInputElements);
    //wbLog(TRACE, "The number of output elements in the input is ", numOutputElements);

    //@@ Allocate GPU memory here
    cudaMalloc(&deviceInput, sizeof(float) * numInputElements);
    cudaMalloc(&deviceOutput, sizeof(float) * numOutputElements);



    //@@ Copy memory to the GPU here
    cudaMemcpy(deviceInput, hostInput, sizeof(float) * numInputElements, cudaMemcpyHostToDevice);


    //@@ Initialize the grid and block dimensions here
    dim3 dimGrid(numOutputElements, 1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);


    //@@ Launch the GPU Kernel here
    sumReduction<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, numInputElements);

    cudaDeviceSynchronize();


 
    //@@ Copy the GPU memory back to the CPU here
    cudaMemcpy(hostOutput, deviceOutput, sizeof(float) * numOutputElements, cudaMemcpyDeviceToHost);



    /********************************************************************
     * Reduce output vector on the host
     * NOTE: One could also perform the reduction of the output vector
     * recursively and support any size input. For simplicity, we do not
     * require that for this lab.
     ********************************************************************/
    for (ii = 1; ii < numOutputElements; ii++) 
    {
        hostOutput[0] += hostOutput[ii];
    }

    printf("Computed answer %f\n", hostOutput[0]);

 
    //@@ Free the GPU memory here
    cudaFree(deviceInput);
    cudaFree(deviceOutput);



    free(hostInput);
    free(hostOutput);

    return 0;
}
