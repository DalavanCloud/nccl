include ../../makefiles/common.mk

.PHONY: all clean

BUILDDIR := $(abspath ../../build)
NVCUFLAGS += -I$(BUILDDIR)/include/ -I../include
ifeq ($(MPI), 1)
NVCUFLAGS += -DMPI_SUPPORT -I$(MPI_HOME)/include
NVLDFLAGS += -L$(MPI_HOME)/lib -lmpi
endif
LIBRARIES += cuda curand nccl nvToolsExt
NVLDFLAGS   += -L$(BUILDDIR)/lib $(LIBRARIES:%=-l%)

DST_DIR := $(BUILDDIR)/test/perf
SRC_FILES := $(wildcard *.cu)
OBJ_FILES := $(SRC_FILES:%.cu=${DST_DIR}/%.o)
BIN_FILES_LIST := all_reduce all_gather broadcast reduce_scatter reduce
BIN_FILES := $(BIN_FILES_LIST:%=${DST_DIR}/%_perf)
ifeq ($(MPI), 1)
BIN_FILES += $(BIN_FILES_LIST:%=${DST_DIR}/%_perf_mpi_transport)
endif
DEP_FILES := $(OBJ_FILES:%.o=%.dep)

all: ${BIN_FILES}

clean:
	rm -rf ${DST_DIR}

ifneq (clean,${MAKECMDGOALS})
-include $(DEP_FILES)
${DST_DIR}/%.dep: %.cu
	@mkdir -p ${DST_DIR}
	$(NVCC) --generate-dependencies $(NVCUFLAGS) $< \
	| sed "0,/^$*.o :/s//$(subst /,\/,${DST_DIR}/$*.o) :/" \
	> $@
endif

${DST_DIR}/%.o: %.cu
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) -c $<

${DST_DIR}/common_mpi.o:common.cu
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -DMPI_TRANSPORT -o $@ $(NVCUFLAGS) -c $<

${DST_DIR}/nccl_mpi.o: ../../share/nccl_mpi.c
	@printf "Compiling   %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	mpicc -std=c99 $(CXXFLAGS) -I$(BUILDDIR)/include/ -o $@ -c $<

${DST_DIR}/%_perf:${DST_DIR}/%.o ${DST_DIR}/common.o
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) $^ ${NVLDFLAGS}

${DST_DIR}/%_perf_mpi_transport: ${DST_DIR}/%.o ${DST_DIR}/common_mpi.o ${DST_DIR}/nccl_mpi.o
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) $^ ${NVLDFLAGS}

