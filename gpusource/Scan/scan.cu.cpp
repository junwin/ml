#include    <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            return -1;                                     \
        }                                                  \
    } while(0)

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

    return 0;
}

