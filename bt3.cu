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

// Kernel 1 - warp divergence
__global__ void reduceByDevice1(int * in, int * out, int n)
{
    // TODO
    int i = (blockIdx.x*blockDim.x + threadIdx.x)*2;

    int stride; 
    int sum = 0;
    for(stride = 0; stride < 2*threadIdx.x; stride *= 2){
        if(threadIdx.x%stride == 0){
            if(i+stride < n){
                sum += in[i+stride];
            }
        }
        __syncthreads();
    }
    out[i] = sum;
    if(threadIdx.x == 0){
        out[blockIdx.x*blockDim.x] = out[blockIdx.x*blockDim.x*2];
    }
}

// Kernel 2 - less warp divergence
__global__ void reduceByDevice2(int * in, int * out, int n)
{
    // TODO
    int num = blockIdx.x*blockDim.x*2;

    for(int stride = 1; stride < blockDim.x*2; stride *= 2){
        int i = num + threadIdx.x*2*stride;
        if(threadIdx.x < blockDim.x/stride){
            if(i + stride < n){
                in[i] += in[i+stride];
            }
            __syncthreads();
        }
    }
    if(threadIdx.x == 0){
        out[blockIdx.x] = in[num];
    }
}

// Kernel 3 - less warp divergence + efficient memory access
__global__ void reduceByDevice3(int * in, int * out, int n)
{
    // TODO
    int num = blockIdx.x*blockDim.x*2;
    for(int stride = blockDim.x; stride > 0; stride /= 2){
        int i = num + threadIdx.x*2*stride;
        if(threadIdx.x < stride){
            if(i + stride < n){
                in[i] += in[i + stride];
            }
            __syncthreads();
        }
    }
    if(threadIdx.x == 0){
        out[blockIdx.x] = in[num];
    }
}

/*
// Kernel 4 - use shared memmory
__global__ void reduceByDevice4(int * in, int * out, int n){
    // todo
    // mỗi block load dữ liệu từ GMEM(ram) lên SMEM
	// vì khai báo cần 1 kích thước tĩnh nên ở đây giả sử kích thước mỗi block là 256
    __shared__ int blkData[2*256];
    // số phần tử trước block hiện tại
    int num = blockIdx.x*blockDim*2;

    blkData[threadIdx.x]=in[num + threadIdx.x];
    blkData[blockDim.x + threadIdx.x]=in[num + blockDim.x + threadIdx.x];

    __syncthreads();

    // tinh toan voi du lieu luu tai SMem
    for(int stride = blockDim.x; stride > 0; stride/=2){
        if(threadIdx.x < stride){
            blkData[threadIdx.x]+=blkData[threadIdx.x + stride];
        }
        __syncthreads();
    }
    // chep du lieu ve lai GMem
    if(threadIdx.x == 0){
        out[blockIdx.x] = blkData[0];
    }
}

*/

int main(int argc, char ** argv)
{
    // Print out device info
    cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));
    printf("**********GPU info**********\n");
    printf("Name: %s\n", devProv.name); // TODO
    printf("Compute capability: %d\n", devProv.major); // TODO
    printf("Num SMs: %d\n", devProv.multiProcessorCount); // TODO
    printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor); // TODO
    printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor/32 ); // TODO
    printf("****************************\n\n");

    // Set up input size
    int n = (1 << 24) + 1;
    printf("Input size: %d\n", n);

    // Set up execution configuration
    dim3 blockSize(256); // Default
    if (argc == 2) // Get block size from cmd argument
        blockSize.x = atoi(argv[1]);
    dim3 gridSize((n-1)/(2*blockSize.x) + 1); // TODO
    printf("Grid size: %d, block size: %d\n", gridSize.x, blockSize.x);

    // Allocate memories
    size_t bytes = n * sizeof(int);
    int * in = (int *) malloc(bytes);
    int * out = (int *) malloc(gridSize.x * sizeof(int));

    // Set up input data
    for (int i = 0; i < n; i++)
    {
        // Generate a random integer in [0, 255]
        in[i] = (int)(rand() & 0xFF);
    }

    // Reduce on host
    int host_sum = reduceByHost(in, n);
    printf("\n%15s%12s%16s%21s%16s\n", 
        "Function", "Result", "KernelTime(ms)", "Post-kernelTime(ms)", "TotalTime(ms)");
    printf("%15s%12d%16s%21s%16s\n", 
        "reduceByHost", host_sum, "-", "-", "-");
    
    //========================================================
    // Allocate device memories
    int *d_in, *d_out;
    CHECK(cudaMalloc(&d_in, bytes));
    CHECK(cudaMalloc(&d_out, gridSize.x * sizeof(int)));
    
    // Copy data to device memories
    CHECK(cudaMemcpy(d_in, in, bytes, cudaMemcpyHostToDevice));

    // Kernel 1 - warp divergence
    CHECK(cudaMemcpy(d_in, in, bytes, cudaMemcpyHostToDevice));
    GpuTimer timer;
    timer.Start();
    reduceByDevice1<<<gridSize, blockSize>>>(d_in, d_out, n);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());
    timer.Stop();
    float kernelTime = timer.Elapsed();
    CHECK(cudaMemcpy(out, d_out, gridSize.x * sizeof(int), cudaMemcpyDeviceToHost));
    timer.Start();
    int device_sum = 0;
    for (int i = 0; i < gridSize.x; i++) 
        device_sum += out[i];
    timer.Stop();
    float postKernelTime = timer.Elapsed();
    printf("%15s%12d%16.3f%21.3f%16.3f\n", 
        "reduceByDevice1", device_sum, kernelTime, postKernelTime, kernelTime + postKernelTime);
    bool correct1 = (host_sum == device_sum); // Check result
    
    // Reset d_in and d_out
    CHECK(cudaMemcpy(d_in, in, bytes, cudaMemcpyHostToDevice)); // Re-copy input data to d_in
    CHECK(cudaMemset(d_out, 0, gridSize.x * sizeof(int))); // Reset d_out

    // Kernel 2 - less warp divergence
    timer.Start();
    reduceByDevice2<<<gridSize, blockSize>>>(d_in, d_out, n);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());
    timer.Stop();
    kernelTime = timer.Elapsed();
    CHECK(cudaMemcpy(out, d_out, gridSize.x * sizeof(int), cudaMemcpyDeviceToHost));
    timer.Start();
    device_sum = 0;
    for (int i = 0; i < gridSize.x; i++) 
        device_sum += out[i];
    timer.Stop();
    postKernelTime = timer.Elapsed();
    printf("%15s%12d%16.3f%21.3f%16.3f\n", 
        "reduceByDevice2", device_sum, kernelTime, postKernelTime, kernelTime + postKernelTime);
    bool correct2 = (host_sum == device_sum); // Check result
    
    // Reset d_in and d_out
    CHECK(cudaMemcpy(d_in, in, bytes, cudaMemcpyHostToDevice)); // Re-copy input data to d_in
    CHECK(cudaMemset(d_out, 0, gridSize.x * sizeof(int))); // Reset d_out
    
    // Kernel 3 - less warp divergence + efficient memory access
    CHECK(cudaMemcpy(d_in, in, bytes, cudaMemcpyHostToDevice)); // Re-copy input data to d_in
    CHECK(cudaMemset(d_out, 0, gridSize.x * sizeof(int))); // Reset d_out
    timer.Start();
    reduceByDevice3<<<gridSize, blockSize>>>(d_in, d_out, n);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());
    timer.Stop();
    kernelTime = timer.Elapsed();   
    CHECK(cudaMemcpy(out, d_out, gridSize.x * sizeof(int), cudaMemcpyDeviceToHost));
    timer.Start();
    device_sum = 0;
    for (int i = 0; i < gridSize.x; i++) 
        device_sum += out[i];
    timer.Stop();
    postKernelTime = timer.Elapsed();
    printf("%15s%12d%16.3f%21.3f%16.3f\n", 
        "reduceByDevice3", device_sum, kernelTime, postKernelTime, kernelTime + postKernelTime);
    bool correct3 = (host_sum == device_sum); // Check result

	/*
    // Reset d_in and d_out
    CHECK(cudaMemcpy(d_in, in, bytes, cudaMemcpyHostToDevice)); // Re-copy input data to d_in
    CHECK(cudaMemset(d_out, 0, gridSize.x * sizeof(int))); // Reset d_out
    
    // Kernel 4 - use shared memory
    CHECK(cudaMemcpy(d_in, in, bytes, cudaMemcpyHostToDevice)); // Re-copy input data to d_in
    CHECK(cudaMemset(d_out, 0, gridSize.x * sizeof(int))); // Reset d_out
    timer.Start();
    reduceByDevice4<<<gridSize, blockSize>>>(d_in, d_out, n);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());
    timer.Stop();
    kernelTime = timer.Elapsed();   
    CHECK(cudaMemcpy(out, d_out, gridSize.x * sizeof(int), cudaMemcpyDeviceToHost));
    timer.Start();
    device_sum = 0;
    for (int i = 0; i < gridSize.x; i++) 
        device_sum += out[i];
    timer.Stop();
    postKernelTime = timer.Elapsed();
    printf("%15s%12d%16.3f%21.3f%16.3f\n", 
        "reduceByDevice4", device_sum, kernelTime, postKernelTime, kernelTime + postKernelTime);
    bool correct4 = (host_sum == device_sum); // Check result
	*/
	
    // Print out errors
    printf("\n");
    if (correct1 == false)
    	fprintf(stderr, "Error: reduceByDevice1 is incorrect!\n");
    if (correct2 == false)
    	fprintf(stderr, "Error: reduceByDevice2 is incorrect!\n");
    if (correct3 == false)
    	fprintf(stderr, "Error: reduceByDevice3 is incorrect!\n");
    //if (correct4 == false)
    //	fprintf(stderr, "Error: reduceByDevice3 is incorrect!\n");
    
        // Free device memories
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    //========================================================
       
    // Free memories
    free(in);
    free(out);
    
    return EXIT_SUCCESS;
}