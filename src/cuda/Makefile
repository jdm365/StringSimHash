# Flags for compiling with NVCC
NVCC_FLAGS = -O3 -arch=sm_80 -Xcompiler

# Flags for pybind11 and Python
PYBIND11_FLAGS = `python3 -m pybind11 --includes` -shared -std=c++11 `python3-config --extension-suffix`

# Include directories for Python
INCLUDES = -I/home/jdm365/miniconda3/include/python3.11

# Linking flags
LD_FLAGS = -L/home/jdm365/miniconda3/lib -lpython3.11

# Source files and output module name
SRCS = jaccard_kernel_bindings.cpp
MODULE_NAME = cuda_jaccard_similarity

all:
	nvcc $(NVCC_FLAGS) $(INCLUDES) $(LD_FLAGS) $(PYBIND11_FLAGS) $(SRCS) -o $(MODULE_NAME)`python3-config --extension-suffix`

run: all

clean:
	rm -f *.o *`python3-config --extension-suffix`
