include ../../makefiles/common.mk
##
BUILDDIR ?= $(abspath ../../build)
### mpi lib
MPI_HOME ?= $(realpath $(dir $(or \
	$(realpath /usr/include/mpi.h),\
	$(realpath /usr/lib/openmpi/include/mpi.h)))..)
ifeq ($(MPI),1)
NVLDFLAGS += -L$(MPI_HOME)/lib -lmpi
NVCUFLAGS += -I$(MPI_HOME)/include
endif
# We need private functions. Include and link with src
NVCUFLAGS += -I$(BUILDDIR)/include/ -I../include -I../../src/include
LIBRARIES += cuda #ibverbs loaded dynamically
LIBOBJS := $(BUILDDIR)/obj/*.o $(BUILDDIR)/obj/*/*.o
NVLDFLAGS += -L$(BUILDDIR)/lib $(LIBRARIES:%=-l%)
##
DST_DIR := $(abspath ${BUILDDIR}/test/unit)
#CU_FILES := $(wildcard *.cu)
CU_FILES := rings_test.cu
ifeq ($(MPI),1)
CU_FILES += ncclNet_test.cu ncclNet_read_write_stress_test.cu
endif
OBJ_FILES:= $(CU_FILES:%.cu=${DST_DIR}/%.o)
EXE_FILES:= $(CU_FILES:%.cu=${DST_DIR}/%)
DEP_FILES:= $(CU_FILES:%.cu=${DST_DIR}/%.dep)
#
all:${EXE_FILES}
clean:
	rm -rf ${DST_DIR}
#
ifneq (clean,${MAKECMDGOALS})
-include ${DEP_FILES}
${DST_DIR}/%.dep:%.cu
	@mkdir -p ${DST_DIR}
	$(NVCC) --generate-dependencies $(NVCUFLAGS) $< \
	| sed "0,/^$*.o :/s//$(subst /,\/,${DST_DIR}/$*.o) :/" \
	> $@
endif
#
${DST_DIR}/%.o:%.cu
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) --compile -o $@ $(NVCUFLAGS) $<
#
${DST_DIR}/nccl_mpi.o: ../../share/nccl_mpi.c
	@printf "Compiling   %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	mpicc -std=c99 $(CXXFLAGS) -I$(BUILDDIR)/include/ -o $@ -c $<
#
${DST_DIR}/rings_test: ${DST_DIR}/rings_test.o ${LIBOBJS}
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) $^ ${NVLDFLAGS}
#
${DST_DIR}/ncclNet_test: ${DST_DIR}/ncclNet_test.o ${DST_DIR}/nccl_mpi.o ${LIBOBJS}
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) ${NVLDFLAGS} $^
#
${DST_DIR}/ncclNet_read_write_stress_test: ${DST_DIR}/ncclNet_read_write_stress_test.o ${LIBOBJS}
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) ${NVLDFLAGS} $^
