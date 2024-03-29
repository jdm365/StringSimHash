# cython: language_level=3

# distutils: language = c++
# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp

from libc.stdlib cimport rand, srand, RAND_MAX
from libc.stdint cimport uint64_t, uint32_t, uint8_t
from libc.stdlib cimport malloc, free

from libcpp.vector cimport vector

cimport numpy as np
import numpy as np
np.import_array()

from collections import defaultdict


SEED = 42
NUM_PERM = 16
NGRAM_SIZE = 3

cdef extern from *:
    """
    #include <stdint.h>
    #include <string.h>

    void parallel_hashes(uint32_t* all_signatures, char** all_data, int* lengths, int num_items, int num_perm) {
        #pragma omp parallel for
        for (int item = 0; item < num_items; ++item) {
            char* data = all_data[item];
            uint32_t* signatures = all_signatures + item * num_perm;
            int length = lengths[item];

            for (int idx = 0; idx < num_perm; ++idx) {
                uint32_t hash = (uint32_t) idx;
                uint32_t m = 0x5bd1e995;
                int r = 24;
                uint32_t n;

                n = *((uint32_t*) (data + length - 4));
                n *= m;
                n &= 0xFFFFFFFF;
                n ^= n >> r;
                n *= m;

                hash *= m;
                hash ^= n;

                hash ^= 4;
                hash ^= hash >> 16;
                hash *= 0x85ebca6b;
                hash &= 0xFFFFFFFF;
                hash ^= hash >> 13;
                hash *= 0xc2b2ae35;
                hash &= 0xFFFFFFFF;
                hash ^= hash >> 15;

                signatures[idx] = hash;
            }
        }
    }
    """
    void parallel_hashes(
            uint32_t* all_signatures, 
            char** all_data, 
            int* lengths, 
            int num_items, 
            int num_perm
            )

## Create parallel_hashes interface
def hash_strings_parallel(items, num_perm=NUM_PERM):
    """
    Hashes a list of strings in parallel using OpenMP.
    """
    num_items = len(items)
    cdef np.ndarray[uint32_t, ndim=2] signatures = np.zeros((num_items, num_perm), dtype=np.uint32)
    cdef np.ndarray[int, ndim=1] lengths = np.zeros(num_items, dtype=np.int32)
    cdef list data_list = []
    
    cdef char** all_data = <char**> malloc(sizeof(char*) * num_items)

    for idx in range(num_items):
        data_list.append(items[idx].encode('utf-8'))
        all_data[idx] = data_list[idx]
        lengths[idx]  = len(data_list[idx])


    parallel_hashes(
        &signatures[0, 0],
        all_data,
        &lengths[0],
        num_items,
        num_perm
    )

    ## Free memory
    free(all_data)

    return signatures

cdef extern from *:
    """
    #include <stdint.h>
    #include <string.h>
    #include <omp.h>

    void _shingle_hash(
            uint32_t* hash_signatures, 
            char** all_data, 
            const int* lengths, 
            int num_items, 
            int n_gram_size
            ) {
        // Calculate the offsets before parallelization
        int offsets[num_items];
        int offset = 0;
        for (int item = 0; item < num_items; ++item) {
            offsets[item] = offset;
            offset += lengths[item];
        }

        // Loop over each item
        #pragma omp parallel for
        for (int item = 0; item < num_items; ++item) {
            char* data = all_data[item];
            int local_offset = offsets[item];

            if (lengths[item] == 1) {
                hash_signatures[local_offset] = 0;
                continue;
            }

            // Loop over each n_gram idx
            for (int idx = 0; idx <= lengths[item]; ++idx) {
                uint32_t hash = 0;
                uint32_t m = 0x5bd1e995;
                int r = 24;

                // Hash each n_gram
                for (int j = 0; j < n_gram_size; ++j) {
                    uint32_t n = (uint32_t)data[idx + j];
                    n *= m;
                    n &= 0xFFFFFFFF;
                    n ^= n >> r;
                    n *= m;

                    hash *= m;
                    hash ^= n;
                }

                // Finalize hash
                hash ^= n_gram_size;
                hash ^= hash >> 16;
                hash *= 0x85ebca6b;
                hash &= 0xFFFFFFFF;
                hash ^= hash >> 13;
                hash *= 0xc2b2ae35;
                hash &= 0xFFFFFFFF;
                hash ^= hash >> 15;

                // Write hash to results array
                hash_signatures[local_offset + idx] = hash;
            }
        }
    }
    """
    void _shingle_hash(
            uint32_t* hash_signatures, 
            char** all_data, 
            int* lengths, 
            int num_items, 
            int n_gram_size,
            )

## Create parallel_hashes interface
def shingle_hash(items, n_gram_size=NGRAM_SIZE):
    """
    Hashes a list of strings in parallel using OpenMP.
    """
    ## signatures should be a list of list containing the signatures for each item
    num_items = len(items)
    cdef np.ndarray[int, ndim=1] lengths = np.zeros(num_items, dtype=np.int32)
    for idx in range(num_items):
        lengths[idx] = max(len(items[idx]) - n_gram_size + 1, 1)

    cdef int total_values = np.sum(lengths)

    ## Create signatures array. 1d array. Determine size by summing the lengths of each item - n_gram_size + 1
    cdef np.ndarray[uint32_t, ndim=1] hash_signatures = np.zeros(total_values, dtype=np.uint32)

    cdef list data_list = []

    cdef char** all_data = <char**> malloc(sizeof(char*) * num_items)

    for idx in range(num_items):
        data_item = items[idx].encode('utf-8')
        data_list.append(data_item)
        all_data[idx] = <char*> data_item

    _shingle_hash(
        &hash_signatures[0],
        all_data,
        &lengths[0],
        num_items,
        n_gram_size
    )

    ## Copy hash_signatures to a list of lists
    signatures = []
    offset = 0
    for idx in range(num_items):
        signatures.append(hash_signatures[offset:offset + lengths[idx]])
        offset += lengths[idx]

    ## Free memory.
    free(all_data)

    return signatures


## Create parallel_hashes interface
def shingle_hash_set(items, n_gram_size=NGRAM_SIZE):
    """
    Hashes a list of strings in parallel using OpenMP.
    """
    ## signatures should be a list of list containing the signatures for each item
    num_items = len(items)
    cdef np.ndarray[int, ndim=1] lengths = np.zeros(num_items, dtype=np.int32)
    for idx in range(num_items):
        lengths[idx] = max(len(items[idx]) - n_gram_size + 1, 1)

    cdef int total_values = np.sum(lengths)

    ## Create signatures array. 1d array. Determine size by summing the lengths of each item - n_gram_size + 1
    cdef np.ndarray[uint32_t, ndim=1] hash_signatures = np.zeros(total_values, dtype=np.uint32)

    cdef list data_list = []

    cdef char** all_data = <char**> malloc(sizeof(char*) * num_items)

    for idx in range(num_items):
        data_item = items[idx].encode('utf-8')
        data_list.append(data_item)
        all_data[idx] = <char*> data_item

    _shingle_hash(
        &hash_signatures[0],
        all_data,
        &lengths[0],
        num_items,
        n_gram_size
    )

    ## Copy hash_signatures to a list of lists
    signatures = []
    offset = 0
    for idx in range(num_items):
        signature = np.unique(hash_signatures[offset:offset + lengths[idx]])
        signatures.append(signature)
        ## offset += lengths[idx]
        offset += len(signature)

    ## Free memory.
    free(all_data)

    return signatures

cdef extern from *:
    """
    #include <stdint.h>
    #include <string.h>

    void parallel_minhash(
        uint32_t* all_signatures, 
        char** all_data, 
        int* lengths, 
        int num_items, 
        int num_perm, 
        int n
        ) {
        uint64_t MERSENNE_PRIME = ((1 << 61) - 1);
        uint64_t MAX_HASH = ((1 << 32) - 1);

        // Generate permutations
        uint64_t* permutations = (uint64_t*) malloc(sizeof(uint64_t) * num_perm * 2);
        for (int idx = 0; idx < num_perm * 2; ++idx) {
            permutations[idx] = rand() % MERSENNE_PRIME;
        }

        #pragma omp parallel for
        for (int item = 0; item < num_items; ++item) {
            char* data = all_data[item];
            uint32_t* signatures = all_signatures + item * num_perm;
            int length = lengths[item];
            for (int idx = 0; idx < num_perm; ++idx) {
                uint32_t min_hash = UINT_MAX;

                for (int shingle_start = 0; shingle_start <= length - n; ++shingle_start) {
                    uint32_t hash = (uint32_t)idx;
                    uint32_t m = 0x5bd1e995;
                    int r = 24;
                    uint32_t n_val;

                    // Hashing n-grams
                    for (int jdx = 0; jdx < n; ++jdx) {
                        n_val = (uint32_t)data[shingle_start + jdx];
                        n_val *= m;
                        n_val &= 0xFFFFFFFF;
                        n_val ^= n_val >> r;
                        n_val *= m;

                        hash *= m;
                        hash ^= n_val;

                        hash ^= 1;  // XOR with the size of one char, which is 1
                    }
                    
                    // Final hash calculations
                    hash ^= n;
                    hash ^= hash >> 16;
                    hash *= 0x85ebca6b;
                    hash &= 0xFFFFFFFF;
                    hash ^= hash >> 13;
                    hash *= 0xc2b2ae35;
                    hash &= 0xFFFFFFFF;
                    hash ^= hash >> 15;

                    // Permutation
                    hash = (hash * permutations[idx * 2]) + permutations[idx * 2 + 1] % MERSENNE_PRIME;
                    hash = hash & MAX_HASH;

                    if (hash < min_hash) {
                        min_hash = hash;
                    }
                }
                signatures[idx] = min_hash;
            }
        }
        free(permutations);
    }
    """
    void parallel_minhash(
            uint32_t* all_signatures, 
            char** all_data, 
            int* lengths, 
            int num_items, 
            int num_perm,
            int ngram_size
            )


cdef extern from *:
    """
    #include <stdint.h>
    #include <string.h>

    void band_hash(
        uint32_t* minhash_signatures,
        int band_size,
        int num_perm,
        int num_items,
        uint64_t* band_hashes
        ) {
        #pragma omp parallel for
        for (int idx = 0; idx < num_items * num_perm / band_size; ++idx) {
            uint32_t signature = minhash_signatures[idx];
            uint64_t band_hash = 0;
            for (int band = 0; band < num_perm / band_size; ++band) {
                band_hash += signature % band_size;
                band_hash *= 1000003;
                signature /= band_size;
            }
            band_hashes[idx] = band_hash;
        }
    }
    """
    void band_hash(
            uint32_t* minhash_signatures,
            int band_size,
            int num_perm,
            int num_items,
            uint64_t* band_hashes
            )


def get_minhashes(items, int num_perm=NUM_PERM):
    cdef int num_items = len(items)
    cdef np.ndarray[np.uint32_t, ndim=2] signatures = np.zeros((num_items, num_perm), dtype=np.uint32)
    cdef np.ndarray[int, ndim=1] lengths = np.zeros(num_items, dtype=np.int32)
    cdef list data_list = []
    
    cdef char** all_data = <char**> malloc(sizeof(char*) * num_items)

    for idx in range(num_items):
        data_list.append(items[idx].encode('utf-8'))
        all_data[idx] = data_list[idx]
        lengths[idx]  = len(data_list[idx])

    ## Get signature permutations using Mersenne's Prime
    MERSENNE_PRIME = np.uint64((1 << 61) - 1)
    permutations = np.array([
        np.random.randint(1, MERSENNE_PRIME, size=num_perm, dtype=np.uint64),
        np.random.randint(0, MERSENNE_PRIME, size=num_perm, dtype=np.uint64)
    ])

    parallel_minhash(&signatures[0, 0], all_data, &lengths[0], num_items, num_perm, NGRAM_SIZE)

    free(all_data)
    return signatures
