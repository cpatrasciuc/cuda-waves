#include <cstdio>

#include <cuda.h>
#include <cuda_runtime_api.h>

__global__ void add(int a, int b, int *c)
{
    *c = a + b;
}

int main()
{
    int c;
    int *dev_c;
    int r;

    cudaMalloc((void**)&dev_c, sizeof(int));

    add<<<1,1>>>(3, 5, dev_c);

    cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);

    printf("%d\n", c);
    
    cudaFree(dev_c);
}
