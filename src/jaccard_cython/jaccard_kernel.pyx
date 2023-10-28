# cython: language_level=3

# distutils: language = c++
# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp

cimport cython

from libc.stdlib cimport rand, srand, RAND_MAX
from libc.stdint cimport uint64_t, uint32_t, uint8_t
from libcpp cimport bool
from libc.stdlib cimport malloc, free
from cpython.array cimport array
from cython.view cimport array as cvarray
from typing import List

from libcpp.vector cimport vector

cimport numpy as np
import numpy as np
np.import_array()



cdef extern from *:
    """
    #include <stdint.h>
    #include <string.h>
    #include <unordered_set>
    #include <vector>
    #include <omp.h>

    inline float jaccard(
        const std::unordered_set<uint32_t>& src_set,
        uint32_t* dst,
        int n_src, 
        int n_dst,
        bool distance
        ) {
        int intersection = 0;
        int union_ = n_src + n_dst;  // Start with the sum of both sets

        // No need to populate src_set; it's already populated

        // Iterate through dst to find intersections and adjust union count
        for (int jdx = 0; jdx < n_dst; ++jdx) {
            if (src_set.find(dst[jdx]) != src_set.end()) {
                intersection++;
                union_--;  // Remove one from union for each intersection
            }
        }

        if (distance) {
            return 1.0f - ((float)intersection / (float)union_);
        }

        return (float)intersection / (float)union_;
    }

    void _cdist_jaccard(
        uint32_t* src, 
        uint32_t* dst, 
        float* result, 
        int n_src, 
        int n_dst, 
        int* src_row_lengths,
        int* dst_row_lengths,
        bool distance
        ) {

        int src_offset = 0;
        int dst_offset = 0;

        // Create a std::unordered_set for each src row
        std::vector<std::unordered_set<uint32_t>> src_sets(n_src);
        for (int idx = 0; idx < n_src; ++idx) {
            src_sets[idx].reserve(src_row_lengths[idx]);
            for (int i = 0; i < src_row_lengths[idx]; ++i) {
                src_sets[idx].insert(src[src_offset + i]);
            }
            src_offset += src_row_lengths[idx];
        }

        #pragma omp parallel for schedule(static)
        for (int idx = 0; idx < n_src; ++idx) {
            dst_offset = 0;
            for (int jdx = 0; jdx < n_dst; ++jdx) {
                result[idx * n_dst + jdx] = jaccard(
                    src_sets[idx],
                    dst + dst_offset,
                    src_row_lengths[idx], 
                    dst_row_lengths[jdx],
                    distance
                );
                dst_offset += dst_row_lengths[jdx];
            }
        }
    }
    """
    void _cdist_jaccard(
        uint32_t* src, 
        uint32_t* dst, 
        float* result, 
        int n_src, 
        int n_dst, 
        int* src_row_lengths,
        int* dst_row_lengths,
        bool distance
        )

def cdist_jaccard(src, dst, calculate_distance=False):
    cdef int n_src = len(src)
    cdef int n_dst = len(dst)

    cdef np.ndarray[float, ndim=1] _result;
    _result = np.zeros((n_src * n_dst), dtype=np.float32)
    
    cdef np.ndarray[int, ndim=1] src_row_lengths;
    cdef np.ndarray[int, ndim=1] dst_row_lengths;
    src_row_lengths = np.zeros(n_src, dtype=np.int32)
    dst_row_lengths = np.zeros(n_dst, dtype=np.int32)

    cdef int total_src_elements = 0
    cdef int total_dst_elements = 0

    for idx in range(n_src):
        src_row_lengths[idx] = len(src[idx])
        total_src_elements += len(src[idx])

    for idx in range(n_dst):
        dst_row_lengths[idx] = len(dst[idx])
        total_dst_elements += len(dst[idx])

    ## Define _src and _dst as arrays of uint32_t
    cdef uint32_t* _src = <uint32_t*> malloc(total_src_elements * sizeof(uint32_t));
    cdef uint32_t* _dst = <uint32_t*> malloc(total_dst_elements * sizeof(uint32_t));

    offset = 0
    for idx in range(n_src):
        for jdx in range(len(src[idx])):
            _src[offset + jdx] = src[idx][jdx]
        offset += len(src[idx])

    offset = 0
    for idx in range(n_dst):
        for jdx in range(len(dst[idx])):
            _dst[offset + jdx] = dst[idx][jdx]
        offset += len(dst[idx])

    _cdist_jaccard(
        <uint32_t*> _src, 
        <uint32_t*> _dst, 
        <float*> _result.data, 
        n_src, 
        n_dst, 
        <int*> src_row_lengths.data,
        <int*> dst_row_lengths.data,
        calculate_distance
        )

    ## Create 2d nparray from result
    cdef np.ndarray[float, ndim=2] result;
    result = np.reshape(_result, (n_src, n_dst))

    ## Free memory
    free(_src)
    free(_dst)

    return result
