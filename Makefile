#
# Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
#
# See LICENCE.txt for license information
#
.PHONY : all clean

default : src.build
BUILDDIR ?= $(abspath ./build)
ABSBUILDDIR := $(abspath $(BUILDDIR))
TARGETS := src test fortran debian doc
all:   ${TARGETS:%=%.build}
clean: ${TARGETS:%=%.clean}
fortran.build test.build: src.build
%.build:
	${MAKE} -C $* build BUILDDIR=${ABSBUILDDIR}

%.clean:
	${MAKE} -C $* clean

${BUILDDIR}/%.txt: %.txt
	cp $< $@

deb: src.build ${BUILDDIR}/NCCL-SLA.txt
	${MAKE} -C debian deb

txz: src.build ${BUILDDIR}/NCCL-SLA.txt ${BUILDDIR}/COPYRIGHT.txt
	${MAKE} -C debian txz
