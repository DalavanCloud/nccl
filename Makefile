#
# Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
#
# See LICENCE.txt for license information
#
.PHONY : all clean
.DEFAULT : all
TARGETS := src test fortran debian
all:   ${TARGETS:%=%.build}
clean: ${TARGETS:%=%.clean}
debian.build:fortran.build src.build
fortran.build test.build: src.build
%.build:
	${MAKE} -C $* all

%.clean:
	${MAKE} -C $* clean
