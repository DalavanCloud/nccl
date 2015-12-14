/*
 * Copyright (c) 2015      NVIDIA Corporation.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef COLL_NCCL_DEBUG_H
#define COLL_NCCL_DEBUG_H


extern int mca_coll_nccl_output;

#define NCCL_VERBOSE(level, format, ...) \
    opal_output_verbose(level, mca_coll_nccl_output, format, ## __VA_ARGS__)

#define NCCL_PROFILE

#ifdef NCCL_PROFILE
#include <time.h>
#define NCCL_TIMING(x) clock_gettime(CLOCK_MONOTONIC, &(x));
#define NCCL_TIMING_ACC(x, y, z) \
    {                                               \
        clock_gettime(CLOCK_MONOTONIC, &(y));       \
        z += get_time(x, y);                        \
    }

inline double get_time(struct timespec t1, struct timespec t2)
{
    return t2.tv_sec - t1.tv_sec + (t2.tv_nsec - t1.tv_nsec)/1e9;
}

#else
#define NCCL_TIMING(x)
#define NCCL_TIMING_ACC(x, y, z)
#endif

#endif
