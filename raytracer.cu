#include <stdio.h>

__global__ void hello_world_thread() {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello world %llu\n", index);
}

void hello_world_gpu() {
    const size_t BLOCK_SIZE = 2;
    size_t GRID_SIZE = 2;    
    hello_world_thread<<<GRID_SIZE, BLOCK_SIZE>>>();
    cudaDeviceSynchronize();
}

int main(int argc, char **argv) {
    hello_world_gpu();
}