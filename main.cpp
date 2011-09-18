#include <cstdio>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "util.h"

#define N 10

__global__ void add(int *a, int *b, int *c)
{
    int i = blockIdx.x;
    c[i] = a[i] + b[i];
}

int main(int argc, char **argv)
{
    int a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;

    for (int i = 0; i < N; i++)
    {
        a[i] = i;
        b[i] = i * i;
    }
    printArray(a, N, "A");
    printArray(b, N, "B");

    cudaMalloc((void**)&dev_a, N * sizeof(int));
    cudaMalloc((void**)&dev_b, N * sizeof(int));
    cudaMalloc((void**)&dev_c, N * sizeof(int));

    cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

    add<<<N,1>>>(dev_a, dev_b, dev_c);

    cudaMemcpy(&c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    printArray(c, N, "C");

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}
