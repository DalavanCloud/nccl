#
# Copyright (c) 2014      The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
# Copyright (c) 2015      NVIDIA Corporation.  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

AM_CPPFLAGS = $(common_cuda_CPPFLAGS) $(coll_nccl_CPPFLAGS)

dist_ompidata_DATA = help-mpi-coll-nccl.txt

sources = coll_nccl_component.c coll_nccl_module.c coll_nccl_allreduce.c coll_nccl_misc.c coll_nccl_multicoll.c coll_nccl_reduce.c coll_nccl_bcast.c coll_nccl_module_misc.cpp
cuda_sources = coll_nccl_kernels.cu
# Make the output library in this directory, and name it either
# mca_<type>_<name>.la (for DSO builds) or libmca_<type>_<name>.la
# (for static builds).

if MCA_BUILD_ompi_coll_cuda_DSO
component_noinst =
component_install = mca_coll_nccl.la
else
component_noinst = libmca_coll_nccl.la
component_install =
endif

noinst_LIBRARIES = libmca_coll_nccl_kernels.a dlink.o

mcacomponentdir = $(ompilibdir)
mcacomponent_LTLIBRARIES = $(component_install)
mca_coll_nccl_la_SOURCES = $(sources)
mca_coll_nccl_la_LDFLAGS = -module -avoid-version $(common_cuda_LDFLAGS) $(coll_nccl_LDFLAGS) -L.
mca_coll_nccl_la_LIBADD = $(coll_nccl_LIBS) $(common_cuda_LIBS) -lmca_coll_nccl_kernels dlink.o

noinst_LTLIBRARIES = $(component_noinst)
libmca_coll_nccl_la_SOURCES = $(sources)
libmca_coll_nccl_la_LDFLAGS = -module -avoi-version $(common_cuda_LDFLAGS) $(coll_nccl_LDFLAGS) -L.
libmca_coll_nccl_la_LIBADD = $(coll_nccl_LIBS) $(common_cuda_LIBS) -lmca_coll_nccl_kernels

#libmca_coll_nccl_kernels_a_CC = $(coll_nccl_NVCC)
libmca_coll_nccl_kernels_a_SOURCES = $(cuda_sources)
libmca_coll_nccl_kernels_a_AR = $(coll_nccl_NVCC) -arch=sm_35 --lib -o

.cu.o:
	$(coll_nccl_NVCC) $(coll_nccl_CPPFLAGS) $(CPPFLAGS) $(DEFAULT_INCLUDES) -arch=sm_35  -Xcompiler -fPIC -dc -o $@ $<

dlink.o: libmca_coll_nccl_kernels.a
	$(coll_nccl_NVCC) -arch=sm_35 -Xcompiler -fPIC -dlink libmca_coll_nccl_kernels.a -o dlink.o
