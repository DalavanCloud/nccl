include ../../makefiles/common.mk
##
BUILDDIR := $(abspath ../../build)
NVCUFLAGS += -I$(BUILDDIR)/include/ -I../include -DMPI=$(MPI)
ifeq ($(MPI), 1)
NVCUFLAGS += -DMPI=1 -I$(MPI_HOME)/include
NVLDFLAGS += -L$(MPI_HOME)/lib -lmpi
endif
LIBRARIES += cuda curand nvToolsExt nccl
NVLDFLAGS   += -L$(BUILDDIR)/lib $(LIBRARIES:%=-l%)
##
DST_DIR := $(abspath ${BUILDDIR}/test/perf)
CU_FILES := $(wildcard *.cu)
OBJ_FILES:= $(CU_FILES:%.cu=${DST_DIR}/%.o)
EXE_FILES_LIST := all_reduce #all_gather broadcast reduce reduce_scatter
EXE_FILES:= $(EXE_FILES_LIST:%=${DST_DIR}/%_perf)
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
${DST_DIR}/%:${DST_DIR}/%.o ${DST_DIR}/common.o
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) $< ${DST_DIR}/common.o ${NVLDFLAGS}
