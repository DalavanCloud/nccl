#
# Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
#
# See LICENCE.txt for license information
#
.PHONY : all clean

default : build
build : deb.build txz.build

BUILDDIR ?= $(abspath ../build)
ABSBUILDDIR := $(abspath $(BUILDDIR))
TARGETS := deb txz
all:   ${TARGETS:%=%.build}
clean: ${TARGETS:%=%.clean}

%.prep:
	${MAKE} -C $* prep BUILDDIR=${ABSBUILDDIR}

%.build:
	${MAKE} -C $* build BUILDDIR=${ABSBUILDDIR}

%.clean:
	${MAKE} -C $* clean
