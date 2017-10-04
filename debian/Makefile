#### PACKAGING ####
include ../makefiles/common.mk
include ../makefiles/version.mk
BUILDDIR ?= $(abspath ../build)
DEBIANDIR  := $(BUILDDIR)/debian

DEBGEN_IN  := $(wildcard *.in)
DEBGEN     := $(DEBGEN_IN:.in=)
DEBFILES   := compat copyright libnccl-dev.install libnccl-dev.docs libnccl2.docs rules $(DEBGEN)
DEBTARGETS := $(patsubst %, $(DEBIANDIR)/%, $(DEBFILES))

PKG_REVISION   ?= 1
PKG_TIMESTAMP  := $(shell date -R)
ARCH           := $(shell uname -m)
PKG_ARCH       ?= $(shell uname -m | sed -e "s/x86_64/amd64/g" | sed -e "s/ppc64le/ppc64el/g")
PKG_MULTIARCH  ?= $(shell gcc -print-multiarch)

build : $(DEBTARGETS)

deb : build
	@printf "Building Debian package\n"
	(cd $(BUILDDIR); debuild -eLD_LIBRARY_PATH -uc -us -d -b)
	mkdir -p $(BUILDDIR)/deb/
	mv $(BUILDDIR)/../libnccl*.deb $(BUILDDIR)/deb/

txz : build
	@printf "Building tar.xz package\n"
	(cd $(BUILDDIR); bash debian/create_txz.sh)
	mkdir -p $(BUILDDIR)/txz/
	mv $(BUILDDIR)/../nccl*.txz $(BUILDDIR)/txz/

clean:
	rm -Rf $(DEBIANDIR)

$(DEBIANDIR)/% : %.in
	@printf "Generating %-35s > %s\n" $< $@
	mkdir -p $(DEBIANDIR)
	sed -e "s/\$${nccl:Major}/$(NCCL_MAJOR)/g" \
	    -e "s/\$${nccl:Minor}/$(NCCL_MINOR)/g" \
	    -e "s/\$${nccl:Patch}/$(NCCL_PATCH)/g" \
	    -e "s/\$${nccl:Suffix}/$(NCCL_SUFFIX)/g" \
	    -e "s/\$${cuda:Major}/$(CUDA_MAJOR)/g" \
	    -e "s/\$${cuda:Minor}/$(CUDA_MINOR)/g" \
	    -e "s/\$${pkg:Revision}/$(PKG_REVISION)/g" \
	    -e "s/\$${pkg:Timestamp}/$(PKG_TIMESTAMP)/g" \
	    -e "s/\$${pkg:Arch}/$(PKG_ARCH)/g" \
	    -e "s/\$${pkg:MultiArch}/$(PKG_MULTIARCH)/g" \
	    $< > $@

$(DEBIANDIR)/% : %
	@printf "Grabbing  %-35s > %s\n" $< $@
	mkdir -p $(DEBIANDIR)
	cp -f $< $@
