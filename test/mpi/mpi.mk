MPI_HOME ?= $(realpath $(dir $(or \
	$(realpath /usr/include/mpi.h),\
	$(realpath /usr/lib/openmpi/include/mpi.h)))..)
MPI_INC ?= $(MPI_HOME)/include
MPI_LIB ?= $(MPI_HOME)/lib
MPI_CPPFLAGS := -I$(MPI_INC) ${CPPFLAGS} ${GTEST_CPPFLAGS}
MPI_LDFLAGS := -L$(MPI_LIB) -lmpi -lcurand
MPITEST_SRC_DIR := test/mpi
MPITEST_SRC_FILES := $(notdir $(wildcard ${MPITEST_SRC_DIR}/*.cu))
MPITEST_DST_DIR := $(BUILDDIR)/test/mpi
MPITEST_OBJ_FILES := ${MPITEST_DST_DIR}/${MPITEST_SRC_FILES:%.cu=%.o}
MPITEST_DEP_FILES := $(MPITEST_OBJ_FILES:%.o=%.dep)
MPITEST_BIN    := ${MPITEST_DST_DIR}/mpi_test
-include $(MPITEST_DEP_FILES)
${MPITEST_DST_DIR}/%.dep: ${MPITEST_SRC_DIR}/%.cu ${INCTARGETS}
	mkdir -p ${MPITEST_DST_DIR} && \
	$(NVCC) -M $(TSTINC) ${MPI_CPPFLAGS} $(NVCUFLAGS) --compiler-options "$(CXXFLAGS)" $< \
	| sed "0,/^$*.o :/s//$(subst /,\/,${MPITEST_DST_DIR}/$*.o) :/" \
	> $@
.PHONY: mpitest mpitest.clean
mpitest : ${MPITEST_BIN}
mpitest.clean:
	rm -rf ${MPITEST_OBJ_FILES} ${MPITEST_BIN}
${MPITEST_BIN}:${GTEST_LIBS} ${MPITEST_OBJ_FILES}
	${NVCC} -o $@ $(TSTLIB) ${MPI_LDFLAGS} $(if ${DEBUG},-Xcompiler --coverage) $^
${MPITEST_OBJ_FILES}:$(TSTDEP)
${MPITEST_DST_DIR}/%.o: ${MPITEST_SRC_DIR}/%.cu ${INCTARGETS}
	$(NVCC) -o $@ --compile $(TSTINC) ${MPI_CPPFLAGS} $(NVCUFLAGS) --compiler-options "$(CXXFLAGS)" $<
#EOF
