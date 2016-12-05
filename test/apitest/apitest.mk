APITEST_SRC_DIR		:= test/apitest
APITEST_SRC_FILES	:= $(notdir $(wildcard ${APITEST_SRC_DIR}/*.cu))
APITEST_DST_DIR		:= $(BUILDDIR)/test/apitest
APITEST_OBJ_FILES	:= $(addprefix ${APITEST_DST_DIR}/,$(APITEST_SRC_FILES:%.cu=%.o))
APITEST_DEP_FILES	:= $(APITEST_OBJ_FILES:%.o=%.dep)
APITEST_CPPFLAGS := $(NVCUFLAGS) -I${NCCL_INC} -include nccl.h ${CPPFLAGS} ${GTEST_CPPFLAGS}
###
.PHONY:apitest apitest.clean apitest.run

-include ${APITEST_DEP_FILES}
${APITEST_DST_DIR}/%.dep: ${APITEST_SRC_DIR}/%.cu ${INCTARGETS}
	mkdir -p ${APITEST_DST_DIR} && \
	$(NVCC) -M ${APITEST_CPPFLAGS} --compiler-options "$(CXXFLAGS)" $< \
	| sed "0,/^$*.o :/s//$(subst /,\/,${APITEST_DST_DIR}/$*.o) :/" \
	> $@

apitest:${APITEST_DST_DIR}/apitest

apitest.clean:
	rm -rf ${APITEST_DST_DIR}

apitest.run:${APITEST_DST_DIR}/apitest
	/bin/bash -c "LD_LIBRARY_PATH+=:$(LIBDIR) $^"

${APITEST_DST_DIR}/apitest:${GTEST_LIBS} ${APITEST_OBJ_FILES} $(LIBDIR)/$(LIBTARGET)
	${NVCC} -o $@ $(TSTLIB) ${GTEST_LIBS} $(if ${DEBUG},-Xcompiler --coverage) ${APITEST_OBJ_FILES}

${APITEST_DST_DIR}/%.o: ${APITEST_SRC_DIR}/%.cu ${INCTARGETS}
	$(NVCC) -o $@ ${APITEST_CPPFLAGS} --compile --compiler-options "$(CXXFLAGS)" $<
