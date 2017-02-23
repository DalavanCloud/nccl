.DEFAULT_GOAL: build
DIRS := apitest single unit perf
ifeq ($(MPI),1)
DIRS += mpi
endif
build: ${DIRS:%=%.build}
clean: ${DIRS:%=%.clean}

%.build:gtest
	${MAKE} -C $*

%.clean:
	${MAKE} -C $* clean

##
include gtest.mk
