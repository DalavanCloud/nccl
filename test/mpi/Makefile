.DEFAULT_GOAL=all
include ../../makefiles/common.mk
BUILDDIR ?= $(abspath ../../build)
GTEST_DIR ?= $(abspath ../googletest/googletest)
include ../gtest.mk
### mpi lib
MPI_HOME ?= $(realpath $(dir $(or \
	$(realpath /usr/include/mpi.h),\
	$(realpath /usr/lib/openmpi/include/mpi.h)))..)
NVLDFLAGS += -L$(MPI_HOME)/lib -lmpi
NVCUFLAGS += -I$(MPI_HOME)/include
### flags
NVCUFLAGS += -I${BUILDDIR}/include/ -include nccl.h -I../include ${CPPFLAGS} ${GTEST_CPPFLAGS}
LIBRARIES += curand nccl
LIBDIR    := ../../build/lib
NVLDFLAGS   += -L${LIBDIR} $(LIBRARIES:%=-l%) $(BUILDDIR)/test/gtest.a
### files
CU_FILES := $(wildcard *.cu)
# CPP_FILES := $(wildcard *.cpp)
DST_DIR := $(BUILDDIR)/test/mpi
CUOBJ_FILES := ${CU_FILES:%.cu=${DST_DIR}/%.o}
# CPPOBJ_FILES := ${CPP_FILES:%.cu=${DST_DIR}/%.o}
DEP_FILES := $(CUOBJ_FILES:%.o=%.dep)
BIN    := ${DST_DIR}/mpi_test
###
ifneq (clean,${MAKECMDGOALS})
-include $(DEP_FILES)
${DST_DIR}/%.dep: %.cu
	@mkdir -p ${DST_DIR}
	$(NVCC) --generate-dependencies $(NVCUFLAGS) $< \
	| sed "0,/^$*.o :/s//$(subst /,\/,${DST_DIR}/$*.o) :/" \
	> $@
endif
#
.PHONY: all clean
all : ${BIN}
#
clean:
	rm -rf ${DST_DIR}
#
${BIN} : ${CUOBJ_FILES}
	${NVCC} -o $@ ${NVLDFLAGS} $^
#
${DST_DIR}/%.o: %.cu
	$(NVCC) -o $@ --compile $(NVCUFLAGS) $<
#
NVIDIASMI := $(or \
	$(realpath /usr/bin/nvidia-smi),\
	$(realpath /usr/local/nvidia/bin/nvidia-smi))
GPU_COUNT := $(shell ${NVIDIASMI} -L|wc -l)
test: ${BIN}
	cd ${BUILDDIR} && \
	export LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:./lib NCCL_DEBUG=WARN && \
	mpiexec -mca btl self,sm -np ${GPU_COUNT} ./test/mpi/mpi_test
#EOF
