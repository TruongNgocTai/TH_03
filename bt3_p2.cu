#include <stdio.h>

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(EXIT_FAILURE);                                                               \
    }                                                                          \
}

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

int reduceByHost(int * in, int n)
{    
    int s = in[0];
    for (int i = 1; i < n; i++)
        s += in[i];
    return s;
}

// Reduce within each block
// Choose the best kernel you have implemented in "bt3.cu"
__global__ void reduceBlksByDevice(int * in, int * out, int n)
{
    // TODO
    
}


// Reduce fully by device
int reduceByDevice(int * in, int n, dim3 blockSize)
{
    // TODO
    
    return 0;
}

// Reduce by device and host
int reduceByDeviceHost(int * in, int n, dim3 blockSize)
{
    // Allocate device memories
    int *d_in, *d_out;
    int bytes = n * sizeof(int);
    CHECK(cudaMalloc(&d_in, bytes));
    dim3 gridSize((n - 1) / (2 * blockSize.x) + 1);
    CHECK(cudaMalloc(&d_out, gridSize.x * sizeof(int)));
    
    // Copy data to device memories
    CHECK(cudaMemcpy(d_in, in, bytes, cudaMemcpyHostToDevice));

    // Invoke kernel function
    reduceBlksByDevice<<<gridSize, blockSize>>>(d_in, d_out, n);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());

    // Copy temporary result from device memories
    int * out = (int *)malloc(gridSize.x * sizeof(int));
    CHECK(cudaMemcpy(out, d_out, gridSize.x * sizeof(int), cudaMemcpyDeviceToHost));

    // (Host) Do the remaining work
    int final_sum = 0;
    for (int i = 0; i < gridSize.x; i++) 
        final_sum += out[i];

    // Free device memories
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));

    return final_sum;  
}

int main(int argc, char ** argv)
{
    // Print out device info
    cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));
    printf("**********GPU info**********\n");
    printf("Name: ...\n"); // TODO
    printf("Compute capability: ...\n"); // TODO
    printf("Num SMs: ...\n"); // TODO
    printf("Max num threads per SM: ...\n"); // TODO
    printf("Max num warps per SM: ...\n"); // TODO
    printf("****************************\n\n");

    // Set up input size
    int n = (1 << 24) + 1;
    printf("Input size: %d\n", n);

    // Set up block size
    dim3 blockSize(256); // Default
    if (argc == 2) // Get block size from cmd argument
        blockSize.x = atoi(argv[1]);

    // Set up input data
    size_t bytes = n * sizeof(int);
    int * in = (int *) malloc(bytes);
    for (int i = 0; i < n; i++)
    {
        // Generate a random integer in [0, 255]
        in[i] = (int)(rand() & 0xFF);
    }

    // Reduce by host
    int host_sum = reduceByHost(in, n);
    
    // Reduce by device-host
    printf("\nreduceByDeviceHost ...\n");
    GpuTimer timer;
    timer.Start();
    int devhost_sum = reduceByDeviceHost(in, n, blockSize);
    timer.Stop();
    printf("Time of reduceByDeviceHost: %.3f ms\n", timer.Elapsed());
    if (devhost_sum != host_sum)
        fprintf(stderr, "Error: reduceByDeviceHost is incorrect!\n");

    // Reduce by device
    printf("\nreduceByDevice ...\n");
    timer.Start();
    int dev_sum = reduceByDevice(in, n, blockSize);
    timer.Stop();
    printf("Time of reduceByDevice    : %.3f ms\n", timer.Elapsed());
    if (dev_sum != host_sum)
        fprintf(stderr, "Error: reduceByDevice is incorrect!\n");

    // Free memories
    free(in);
    
    return EXIT_SUCCESS;
}