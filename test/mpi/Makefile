.DEFAULT_GOAL=all
include ../../makefiles/common.mk
BUILDDIR ?= $(abspath ../../build)
GTEST_DIR ?= $(abspath ../googletest/googletest)
include ../gtest.mk
### mpi lib
MPI_HOME ?= $(realpath $(dir $(or \
	$(realpath /usr/include/mpi.h),\
	$(realpath /usr/lib/openmpi/include/mpi.h)))..)
NVLDFLAGS += -L$(MPI_HOME)/lib -lmpi -lmpi_cxx
NVCUFLAGS += -I$(MPI_HOME)/include
### flags
NVCUFLAGS += -I${BUILDDIR}/include/ -include nccl.h -I../include ${CPPFLAGS} ${GTEST_CPPFLAGS}
LIBRARIES += curand nccl
LIBDIR    := ../../build/lib
NVLDFLAGS   += -L${LIBDIR} $(LIBRARIES:%=-l%) $(BUILDDIR)/test/gtest.a
### files
SRC_FILES := $(wildcard *.cpp) $(wildcard *.cu)
DST_DIR := $(BUILDDIR)/test/mpi
OBJ_FILES := ${SRC_FILES:%.cpp=${DST_DIR}/%.o}
OBJ_FILES := ${OBJ_FILES:%.cu=${DST_DIR}/%.o}
DEP_FILES := ${OBJ_FILES:%.o=%.dep}
BIN    := ${DST_DIR}/mpi_test
###
ifneq (clean,${MAKECMDGOALS})
-include $(DEP_FILES)
${DST_DIR}/%.dep: %.cpp
	@mkdir -p ${DST_DIR}
	${NVCC} -M $(NVCUFLAGS) $< \
	| sed "0,/^$*.o :/s//$(subst /,\/,${DST_DIR}/$*.o) :/" \
	> $@
${DST_DIR}/%.dep: %.cu
	@mkdir -p ${DST_DIR}
	${NVCC} -M $(NVCUFLAGS) $< \
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
${BIN} : ${OBJ_FILES}
	${NVCC} -o $@ ${NVLDFLAGS} $^
#
${DST_DIR}/%.o: %.cpp
	${NVCC} -o $@ --compile $(NVCUFLAGS) $<
${DST_DIR}/%.o: %.cu
	${NVCC} -o $@ --compile $(NVCUFLAGS) $<
#
NVIDIASMI := $(or \
	$(realpath /usr/bin/nvidia-smi),\
	$(realpath /usr/local/nvidia/bin/nvidia-smi))
GPU_COUNT := $(shell ${NVIDIASMI} -L|wc -l)
test: ${BIN}
	cd ${BUILDDIR} && \
	export LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:./lib NCCL_DEBUG=WARN && \
	mpiexec -mca btl self,sm -np ${GPU_COUNT} ./test/mpi/mpi_test

perf: ${BIN}
	cd ${BUILDDIR} && \
	export LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:./lib NCCL_DEBUG=WARN && \
	mpiexec -mca btl self,sm -np ${GPU_COUNT} ./test/mpi/mpi_test -perf
#EOF
