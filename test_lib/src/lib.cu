#include <stdio.h>

// Device code

// Host code
bool isSharedMemAllocated = false; 

int main() {

  // Allocate shared memory
  if(!isSharedMemAllocated) {
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
    isSharedMemAllocated = true; 
  }

  // Launch kernel with reused shared memory
  levenshteinKernel<<<128/BLOCK_SIZE, BLOCK_SIZE>>>(strings, numStrings, 
                                                    otherString, distances);

  // Free shared memory  
  if(isSharedMemAllocated) {
    cudaDeviceResetSharedMemConfig();
    isSharedMemAllocated = false;
  }

  return 0;
}
