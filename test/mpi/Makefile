include ../../makefiles/common.mk

.PHONY: all clean

BUILDDIR := $(abspath ../../build)
NVCUFLAGS += -I$(BUILDDIR)/include/ -I../include
ifeq ($(MPI), 1)
NVCUFLAGS += -DMPI_SUPPORT -I$(MPI_HOME)/include
NVLDFLAGS += -L$(MPI_HOME)/lib -lmpi
endif
LIBRARIES += cuda curand nccl
NVLDFLAGS   += -L$(BUILDDIR)/lib $(LIBRARIES:%=-l%)

DST_DIR := $(BUILDDIR)/test/mpi
SRC_FILES := $(wildcard *.cu)
OBJ_FILES := $(SRC_FILES:%.cu=${DST_DIR}/%.o)
BIN_FILES_LIST := mpi_perf all_reduce_scan
BIN_FILES := $(BIN_FILES_LIST:%=${DST_DIR}/%)
ifeq ($(MPI), 1)
BIN_FILES += $(BIN_FILES_LIST:%=${DST_DIR}/%_mpi_transport)
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

${DST_DIR}/nccl_mpi.o: ../../share/nccl_mpi.c
	@printf "Compiling   %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	mpicc -std=c99 $(CXXFLAGS) -I$(BUILDDIR)/include/ -o $@ -c $<

${DST_DIR}/%:${DST_DIR}/%.o
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) $^ ${NVLDFLAGS}

${DST_DIR}/%_mpi_transport: ${DST_DIR}/%.o ${DST_DIR}/nccl_mpi.o
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) $^ ${NVLDFLAGS}

