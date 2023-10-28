#include <cuda_runtime.h>


#define TILE_SIZE 32
#define MAX_SET_SIZE 1024

__constant__ uint32_t constMemM[1024];
__constant__ int startIndicesM[128];

__global__ void jaccard_similarity(
		const uint32_t* setsN, 
		const int* lengthsN, 
		const uint32_t* setsM,
		const int* lengthsM,
		float* result, 
		const int N, 
		const int M
		) {

    int idxM = blockIdx.x;
    int idxN = blockIdx.y * blockDim.y + threadIdx.y;

    if (idxM >= M || idxN >= N) {
        return;
    }

    int lenM = lengthsM[idxM];
    int lenN = lengthsN[idxN];

    __shared__ uint32_t sharedM[TILE_SIZE];
    __shared__ uint32_t sharedN[TILE_SIZE];

    int i = 0, j = 0;
    int intersection_count = 0;
    int union_count = lenM + lenN;

    while (i < lenM && j < lenN) {

        // Loading tile from setM into shared memory
        if (threadIdx.y < TILE_SIZE && i + threadIdx.y < lenM) {
            sharedM[threadIdx.y] = setsM[idxM * MAX_SET_SIZE + i + threadIdx.y];
        }

        // Loading tile from setN into shared memory
        if (threadIdx.y < TILE_SIZE && j + threadIdx.y < lenN) {
            sharedN[threadIdx.y] = setsN[idxN * MAX_SET_SIZE + j + threadIdx.y];
        }

        __syncthreads();

        int local_i = 0, local_j = 0;
        while (local_i < TILE_SIZE && local_j < TILE_SIZE) {
            if (sharedM[local_i] == sharedN[local_j]) {
                ++intersection_count;
                --union_count;
                ++local_i;
                ++local_j;
            } else if (sharedM[local_i] < sharedN[local_j]) {
                ++local_i;
            } else {
                ++local_j;
            }
        }

        __syncthreads();

        if (local_i >= TILE_SIZE) i += TILE_SIZE;
        if (local_j >= TILE_SIZE) j += TILE_SIZE;
    }

    result[idxM * N + idxN] = (float) intersection_count / union_count;
}
