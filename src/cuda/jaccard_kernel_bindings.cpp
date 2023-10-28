#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>

extern __global__ void jaccard_similarity(
		uint32_t* setsN,
		int* lengthsN, 
		uint32_t* setsM, 
		int* lengthsM,
		float* result,
		const int N,
		const int M
		);

extern __constant__ uint32_t constMemM[1024];
extern __constant__ int startIndicesM[128];


pybind11::array_t<float> cuda_jaccard_similarity(
		pybind11::array_t<uint32_t> setsN, 
		pybind11::array_t<int> lengthsN, 
		pybind11::array_t<uint32_t> setsM, 
		pybind11::array_t<int> lengthsM, 
		pybind11::array_t<float> result,
		const int N,
		const int M
		) {
	// Copy data from Python to C++
	
	// Get pointers to the data
	auto ptrN = setsN.mutable_unchecked<2>();
	auto ptrLengthsN = lengthsN.mutable_unchecked<1>();
	auto ptrM = setsM.mutable_unchecked<2>();
	auto ptrLengthsM = lengthsM.mutable_unchecked<1>();
	auto ptrResult = result.mutable_unchecked<2>();

	// Allocate memory on the GPU
	uint32_t* d_setsN;
	int* 	  d_lengthsN;
	uint32_t* d_setsM;
	int* 	  d_lengthsM;
	float* 	  d_result;

	cudaMalloc(&d_setsN, N * 1024 * sizeof(uint32_t));
	cudaMalloc(&d_lengthsN, N * sizeof(int));
	cudaMalloc(&d_setsM, M * 1024 * sizeof(uint32_t));
	cudaMalloc(&d_lengthsM, M * sizeof(int));
	cudaMalloc(&d_result, N * M * sizeof(float));

	// Copy data from C++ to GPU
	cudaMemcpy(d_setsN, ptrN.mutable_data(0, 0), N * 1024 * sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_lengthsN, ptrLengthsN.mutable_data(0), N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_setsM, ptrM.mutable_data(0, 0), M * 1024 * sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_lengthsM, ptrLengthsM.mutable_data(0), M * sizeof(int), cudaMemcpyHostToDevice);

	// Copy data from C++ to GPU constant memory
	cudaMemcpyToSymbol(constMemM, ptrM.mutable_data(0, 0), M * 1024 * sizeof(uint32_t));
	cudaMemcpyToSymbol(startIndicesM, ptrLengthsM.mutable_data(0), M * sizeof(int));

	// Call the kernel
	jaccard_similarity<<<N, M>>>(
			d_setsN, 
			d_lengthsN, 
			d_setsM, 
			d_lengthsM, 
			d_result, 
			N, 
			M
			);

	// Copy data from GPU to C++
	cudaMemcpy(ptrResult.mutable_data(0, 0), d_result, N * M * sizeof(float), cudaMemcpyDeviceToHost);

	// Free memory on the GPU
	cudaFree(d_setsN);
	cudaFree(d_lengthsN);
	cudaFree(d_setsM);
	cudaFree(d_lengthsM);
	cudaFree(d_result);

	return result;
}

PYBIND11_MODULE(your_module_name, m) {
    m.def("cuda_jaccard_similarity", &cuda_jaccard_similarity, "Calculate Jaccard Similarity using CUDA");
}
