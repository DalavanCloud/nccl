.DEFAULT_GOAL=all
include ../../makefiles/common.mk
BUILDDIR ?= $(abspath ../../build)
GTEST_DIR ?= $(abspath ../googletest/googletest)
include ../gtest.mk
### files
SRC_FILES	:= $(wildcard *.cu)
DST_DIR		:= $(BUILDDIR)/test/apitest
OBJ_FILES	:= $(SRC_FILES:%.cu=${DST_DIR}/%.o)
DEP_FILES	:= $(OBJ_FILES:%.o=%.dep)
### cc
NVCUFLAGS += -I${BUILDDIR}/include/ -include nccl.h ${CPPFLAGS} ${GTEST_CPPFLAGS}
### ld
LIBDIR := ${BUILDDIR}/lib
NVLDFLAGS   += -L${LIBDIR} ${GTEST_LIBS}
### rules
.PHONY:

ifneq (clean,${MAKECMDGOALS})
-include ${DEP_FILES}
${DST_DIR}/%.dep:%.cu
	@mkdir -p ${DST_DIR}
	$(NVCC) --generate-dependencies ${NVCUFLAGS} $< \
	| sed "0,/^$*.o :/s//$(subst /,\/,${DST_DIR}/$*.o) :/" \
	> $@
endif
all:${DST_DIR}/apitest ${DST_DIR}/apitest_static

clean:
	rm -rf ${DST_DIR}

test: ${DST_DIR}/apitest
	/bin/bash -c "LD_LIBRARY_PATH+=:$(LIBDIR) $^"

${DST_DIR}/apitest:${OBJ_FILES}
	${NVCC} -o $@ ${NVCUFLAGS} ${NVLDFLAGS} $^ -lnccl

${DST_DIR}/apitest_static:${OBJ_FILES}
	${NVCC} -o $@ ${NVCUFLAGS} ${NVLDFLAGS} $^ -lnccl_static

${DST_DIR}/%.o: %.cu
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ ${NVCUFLAGS} --compile $<
