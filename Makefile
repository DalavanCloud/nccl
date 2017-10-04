#
# Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
#
# See LICENCE.txt for license information
#
.PHONY : all clean

default : src.build
BUILDDIR ?= $(abspath ./build)
ABSBUILDDIR := $(abspath $(BUILDDIR))
TARGETS := src test pkg
clean: ${TARGETS:%=%.clean}
test.build: src.build
LICENSE_FILES := NCCL-SLA.txt COPYRIGHT.txt
LICENSE_TARGETS := $(LICENSE_FILES:%=$(BUILDDIR)/%)
lic: $(LICENSE_TARGETS)

${BUILDDIR}/%.txt: %.txt
	@printf "Copying    %-35s > %s\n" $< $@
	mkdir -p ${BUILDDIR}
	cp $< $@

src.%:
	${MAKE} -C src $* BUILDDIR=${ABSBUILDDIR}

test.%:
	${MAKE} -C test $* BUILDDIR=${ABSBUILDDIR}

pkg.%:
	${MAKE} -C pkg $* BUILDDIR=${ABSBUILDDIR}
