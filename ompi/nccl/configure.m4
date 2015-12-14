# -*- shell-script -*-
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

# MCA_coll_nccl_CONFIG([action-if-can-compile],
#                      [action-if-cant-compile])
# ------------------------------------------------
AC_DEFUN([MCA_ompi_coll_nccl_CONFIG],[
    AC_CONFIG_FILES([ompi/mca/coll/nccl/Makefile])
    # make sure that CUDA-aware checks have been done
    AC_ARG_WITH([nccl],
                [AC_HELP_STRING([--with-nccll(=DIR)],
                                [Build NCCL support, optionally adding DIR/lib and DIR/include. Requires CUDA support (--with-cuda)])])
    AC_MSG_CHECKING([if --with-nccl is set])
    AS_IF([test "x$with_nccl" = "xno" -o "x$with_nccl" = "x"],
          [mca_ompi_coll_nccl_happy="no"
           AC_MSG_RESULT([not set (--with-nccl=$with_nccl)])],
          [AS_IF([test "x$with_nccl" = "xyes"],
                 [AS_IF([test "x`ls /usr/local/include/nccl.h 2> /dev/null`" = "x"],
                        [AC_MSG_RESULT([not found in standard location])
                         AC_MSG_WARN([Expected file /usr/local/include/nccl.h not found])
                         AC_MSG_ERROR([Cannot continue])],
                        [AC_MSG_RESULT([found])
                         mca_ompi_coll_nccl_happy="yes"
                         mca_ompi_coll_nccl_dir=/usr/local])],
                 [AS_IF([test ! -d "$with_nccl"],
                        [AC_MSG_RESULT([not found])
                         AC_MSG_WARN([Directory $with_nccl not found])
                         AC_MSG_ERROR([Cannot continue])],
                        [AS_IF([test "x`ls $with_nccl/include/nccl.h 2> /dev/null`" = "x"],
                               [AC_MSG_RESULT([not found])
                               AC_MSG_WARN([Expected file nccl.h in $with_nccl/include not found])
                               AC_MSG_ERROR([Cannot continue])],
                               [mca_ompi_coll_nccl_happy="yes"
                                mca_ompi_coll_nccl_dir=$with_nccl
                                AC_MSG_RESULT([found ($mca_ompi_coll_nccl_dir/include/nccl.h)])]
                              )]
                       )]
                )]
         )



    AC_REQUIRE([OPAL_CHECK_CUDA])
    AS_IF([test "x$with_cuda" = "xyes"],
          [mca_ompi_coll_nccl_cuda_dir=/usr/local/cuda],
          [mca_ompi_coll_nccl_cuda_dir=$with_cuda])
    AC_MSG_NOTICE([found CUDA directory: $mca_ompi_coll_nccl_cuda_dir])
    AC_PATH_PROG(mca_ompi_coll_nccl_nvcc, [nvcc], [notfound], [$PATH:$mca_ompi_coll_nccl_cuda_dir/bin])


    coll_nccl_LIBS="-lnccl -lcudart -lcuda -lcurand -lnvToolsExt"
    coll_nccl_LDFLAGS="-L$mca_ompi_coll_nccl_dir/lib -L$mca_ompi_coll_nccl_cuda_dir/lib64"
    coll_nccl_CPPFLAGS="-I$mca_ompi_coll_nccl_dir/include"
    coll_nccl_NVCC="$mca_ompi_coll_nccl_nvcc"

    # Only build if CUDA support is available
    AS_IF([test "x$CUDA_SUPPORT" = "x1" -a "x$mca_ompi_coll_nccl_happy" = "xyes" -a "x%mca_ompi_coll_nccl_nvcc" != "xnotfound"],
          [$1],
          [$2])
    AC_SUBST([coll_nccl_LIBS])
    AC_SUBST([coll_nccl_LDFLAGS])
    AC_SUBST([coll_nccl_CPPFLAGS])
    AC_SUBST([coll_nccl_NVCC])

])dnl

