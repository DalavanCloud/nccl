##### version
NCCL_MAJOR   := 2
NCCL_MINOR   := 0
NCCL_PATCH   := 0
CUDA_VERSION ?= $(shell ls $(CUDA_LIB)/libcudart.so.*.*.* | rev | cut -d "." -f -3 | rev)
CUDA_MAJOR = $(shell echo $(CUDA_VERSION) | cut -d "." -f 1)
CUDA_MINOR = $(shell echo $(CUDA_VERSION) | cut -d "." -f 2)
CUDA_BUILD = $(shell echo $(CUDA_VERSION) | cut -d "." -f 3)

CXXFLAGS  += -DCUDA_MAJOR=$(CUDA_MAJOR) -DCUDA_MINOR=$(CUDA_MINOR) -DCUDA_BUILD=$(CUDA_BUILD)
