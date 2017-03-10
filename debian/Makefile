#### PACKAGING ####
include ../makefiles/common.mk
include ../makefiles/version.mk
BUILDDIR ?= $(abspath ../build)
DEBIANDIR  := $(BUILDDIR)/debian

DEBGEN_IN  := $(wildcard *.in)
DEBGEN     := $(DEBGEN_IN:.in=)
DEBFILES   := compat copyright libnccl-dev.install libnccl-dev.manpages nccl.7 rules $(DEBGEN)
DEBTARGETS := $(patsubst %, $(DEBIANDIR)/%, $(DEBFILES))

DEB_REVISION   ?= 1
DEB_TIMESTAMP  := $(shell date -R)
DEB_ARCH       ?= amd64

build : $(DEBTARGETS)

package : build
	@printf "Building Debian package\n"
	(cd $(BUILDDIR); debuild -eLD_LIBRARY_PATH -uc -us -d -b)
	mkdir -p $(BUILDDIR)/deb/
	mv $(BUILDDIR)/../libnccl*.deb $(BUILDDIR)/deb/

clean:
	rm -Rf $(DEBIANDIR)

$(DEBIANDIR)/% : %.in
	@printf "Generating %-35s > %s\n" $< $@
	sed -e "s/\$${nccl:Major}/$(NCCL_MAJOR)/g" \
	    -e "s/\$${nccl:Minor}/$(NCCL_MINOR)/g" \
	    -e "s/\$${nccl:Patch}/$(NCCL_PATCH)/g" \
	    -e "s/\$${cuda:Major}/$(CUDA_MAJOR)/g" \
	    -e "s/\$${cuda:Minor}/$(CUDA_MINOR)/g" \
	    -e "s/\$${deb:Revision}/$(DEB_REVISION)/g" \
	    -e "s/\$${deb:Timestamp}/$(DEB_TIMESTAMP)/g" \
	    -e "s/\$${deb:Arch}/$(DEB_ARCH)/g" \
	    $< > $@

$(DEBIANDIR)/% : %
	@printf "Grabbing  %-35s > %s\n" $< $@
	mkdir -p $(DEBIANDIR)
	cp -f $< $@
