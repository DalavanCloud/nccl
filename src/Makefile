include ../makefiles/common.mk
include ../makefiles/version.mk

##### src files
INCEXPORTS  := nccl.h
LIBSRCFILES := init.cu bootstrap.cu transport.cu misc/group.cu \
		misc/nvmlwrap.cu misc/crc32.cu misc/rings.cu misc/utils.cu misc/gdcopy.cu \
		transport/p2p.cu transport/shm.cu transport/net.cu transport/net_socket.cu \
		collectives/all_gather.cu collectives/all_reduce.cu collectives/broadcast.cu \
		collectives/reduce.cu collectives/reduce_scatter.cu
##### lib files
LIBNAME     := libnccl.so
STATICLIBNAME := libnccl_static.a
##### dirs
BUILDDIR ?= $(abspath ../build)
INCDIR := $(BUILDDIR)/include
LIBDIR := $(BUILDDIR)/lib
OBJDIR := $(BUILDDIR)/obj
##### target files
INCTARGETS := $(INCEXPORTS:%=$(INCDIR)/%)
LIBSONAME  := $(LIBNAME:%=%.$(NCCL_MAJOR))
LIBTARGET  := $(LIBNAME:%=%.$(NCCL_MAJOR).$(NCCL_MINOR).$(NCCL_PATCH))
STATICLIBTARGET := $(STATICLIBNAME)
LIBOBJ     := $(LIBSRCFILES:%.cu=$(OBJDIR)/%.o)
DEPFILES   := $(LIBOBJ:%.o=%.d)

##### rules
build : lib staticlib

lib : $(INCTARGETS) $(LIBDIR)/$(LIBTARGET)

staticlib : $(INCTARGETS) $(LIBDIR)/$(STATICLIBTARGET)

-include $(DEPFILES)
$(LIBDIR)/$(LIBTARGET) $(LIBDIR)/$(STATICLIBTARGET) : $(LIBOBJ)

nccl.h : nccl.h.in
	@printf "Generating %-35s > %s\n" $< $@
	sed -e "s/\$${nccl:Major}/$(NCCL_MAJOR)/g" \
	    -e "s/\$${nccl:Minor}/$(NCCL_MINOR)/g" \
	    -e "s/\$${nccl:Patch}/$(NCCL_PATCH)/g" \
	    $< > $@

$(LIBDIR)/$(LIBTARGET):
	@printf "Linking    %-35s > %s\n" $(LIBTARGET) $@
	mkdir -p $(LIBDIR)
	$(CXX) $(CXXFLAGS) -shared -Wl,--no-as-needed -Wl,-soname,$(LIBSONAME) -o $@ $(LDFLAGS) $(LIBOBJ)
	ln -sf $(LIBSONAME) $(LIBDIR)/$(LIBNAME)
	ln -sf $(LIBTARGET) $(LIBDIR)/$(LIBSONAME)

$(LIBDIR)/$(STATICLIBTARGET):
	@printf "Archiving  %-35s > %s\n" $(STATICLIBTARGET) $@
	mkdir -p $(LIBDIR)
	ar cr $@ $(LIBOBJ)

$(INCDIR)/%.h : %.h
	@printf "Grabbing   %-35s > %s\n" $< $@
	mkdir -p $(INCDIR)
	cp -f $< $@

$(OBJDIR)/%.o : %.cu
	@printf "Compiling  %-35s > %s\n" $< $@
	mkdir -p `dirname $@`
	$(NVCC) -I. -Iinclude -c $(NVCUFLAGS) --compiler-options "$(CXXFLAGS)" $< -o $@
	@$(NVCC) -I. -Iinclude -M $(NVCUFLAGS) --compiler-options "$(CXXFLAGS)" $< > $(@:%.o=%.d.tmp)
	@sed "0,/^.*:/s//$(subst /,\/,$@):/" $(@:%.o=%.d.tmp) > $(@:%.o=%.d)
	@sed -e 's/.*://' -e 's/\\$$//' < $(@:%.o=%.d.tmp) | fmt -1 | \
                sed -e 's/^ *//' -e 's/$$/:/' >> $(@:%.o=%.d)
	@rm -f $(@:%.o=%.d.tmp)

clean :
	rm -rf ${INCDIR} ${LIBDIR} ${OBJDIR} nccl.h

install : lib
	mkdir -p $(PREFIX)/lib
	mkdir -p $(PREFIX)/include
	cp -P -v $(BUILDDIR)/lib/* $(PREFIX)/lib/
	cp -v $(BUILDDIR)/include/* $(PREFIX)/include/

