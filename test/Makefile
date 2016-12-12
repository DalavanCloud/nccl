.DEFAULT_GOAL: all
DIRS := apitest mpi single
all: ${DIRS:%=%.build}
clean: ${DIRS:%=%.clean}

%.build:
	${MAKE} -C $*

%.clean:
	${MAKE} -C $* clean

apitest.build mpi.build single.build: gtest
##
include gtest.mk
