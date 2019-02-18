/*
 * This is a simple CUDA code that negates an array of integers.
 * It introduces the concepts of device memory management, and
 * kernel invocation.
 *
 * Training material developed by James Perry and Alan Gray
 * Copyright EPCC, The University of Edinburgh, 2013 
 */

#include <stdio.h>
#include <stdlib.h>

/* Forward Declaration*/
/* Utility function to check for and report CUDA errors */
void checkCUDAError(const char*);

/* The actual array negation kernel (basic single block version) */
__global__ void negate(int *d_a)
{
    int idx = threadIdx.x;
    d_a[idx] = -1 * d_a[idx];
}

/* Multi-block version of kernel for part 2C */
__global__ void negate_multiblock(int *d_a)
{
    int idx = threadIdx.x + (blockIdx.x * blockDim.x);
    d_a[idx] = -1 * d_a[idx];
}

/* The number of integer elements in the array */
#define ARRAY_SIZE 256

/*
 * The number of CUDA blocks and threads per block to use.
 * These should always multiply to give the array size.
 * For the single block kernel, NUM_BLOCKS should be 1 and
 * THREADS_PER_BLOCK should be the array size
 */
#define NUM_BLOCKS  1
#define THREADS_PER_BLOCK 256

/* Main routine */
int main(int argc, char *argv[])
{
    int *h_a, *h_out;
    int *d_a;

    int i;
    size_t sz = ARRAY_SIZE * sizeof(int);

    /*
     * allocate memory on host
     * h_a holds the input array, h_out holds the result
     */
    h_a = (int *) malloc(sz);
    h_out = (int *) malloc(sz);

    /*
     * allocate memory on device
     */
    cudaMalloc(&d_a, sz);

    /* initialise host arrays */
    for (i = 0; i < ARRAY_SIZE; i++) {
        h_a[i] = i;
        h_out[i] = 0;
    }

    /* copy input array from host to GPU */
    cudaMemcpy(d_a, h_a, sz, cudaMemcpyHostToDevice);

    /* run the kernel on the GPU */
    dim3 blocksPerGrid(NUM_BLOCKS,1,1);
    dim3 threadsPerBlock(THREADS_PER_BLOCK,1,1);
    negate<<< blocksPerGrid, threadsPerBlock >>>(d_a);

    /* wait for all threads to complete and check for errors */
    cudaThreadSynchronize();
    checkCUDAError("kernel invocation");

    /* copy the result array back to the host */
    cudaMemcpy(h_out, d_a, sz, cudaMemcpyDeviceToHost);

    checkCUDAError("memcpy");

    /* print out the result */
    printf("Results: ");
    for (i = 0; i < ARRAY_SIZE; i++) {
      printf("%d, ", h_out[i]);
    }
    printf("\n\n");

    /* free device buffer */
    cudaFree(d_a);

    /* free host buffers */
    free(h_a);
    free(h_out);

    return 0;
}


/* Utility function to check for and report CUDA errors */
void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) 
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }                         
}
