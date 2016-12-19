.DEFAULT_GOAL: all
DIRS := apitest mpi single
all: ${DIRS:%=%.build}
clean: ${DIRS:%=%.clean}

%.build:gtest
	${MAKE} -C $*

%.clean:
	${MAKE} -C $* clean

##
include gtest.mk
