/*
 * Copyright (c) 2015 NVIDIA Corporation.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifdef __cplusplus
extern "C"
#endif
void mca_coll_nccl_populate_rankinfo(int * hosts, int * nccl_ranks,
                                     int * intercomm_ranks, int * leader, int * intercolor,
                                     int my_color, int my_rank, int size);
