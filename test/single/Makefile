include ../../makefiles/common.mk
##
BUILDDIR := $(abspath ../../build)
NVCUFLAGS += -I../../build/include/ -I../include
LIBRARIES += cuda curand nvToolsExt nccl
NVLDFLAGS   += -L../../build/lib $(LIBRARIES:%=-l%)
##
DST_DIR := $(abspath ${BUILDDIR}/test/single)
CU_FILES := $(wildcard *.cu)
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
${DST_DIR}/%:${DST_DIR}/%.o
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) $< ${NVLDFLAGS}
