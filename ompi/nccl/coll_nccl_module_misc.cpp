/*
 * Copyright (c) 2015 NVIDIA Corporation.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include <map>
#include <cstdio>
#include "coll_nccl_module_misc.h"

using namespace std;

extern "C"
void mca_coll_nccl_populate_rankinfo(int * hosts, int * nccl_ranks,
                                     int * intercomm_ranks, int * leader, int * intercolor,
                                     int my_color, int my_rank, int size)
{
    map<int, int> color_map;
    *intercolor = 0;
    for(int i = 0; i < size; ++i)
    {
        if(i < my_rank)
        {
            if(hosts[i] == my_color)
            {
                if(*intercolor == 0)
                    *leader = i;
                ++(*intercolor);
            }
        }
        map<int,int>::iterator it = color_map.find(hosts[i]);
        if(it == color_map.end())
        {
            color_map[hosts[i]] = 0;
            nccl_ranks[i] = 0;
        }
        else
        {
            nccl_ranks[i] = ++color_map[hosts[i]];
        }
    }
    if(*intercolor == 0)
        *leader = my_rank;
    int count = 0;
    for(int i = 0; i < size; ++i)
    {
        if(nccl_ranks[i] == *intercolor)
        {
            intercomm_ranks[i] = count;
            ++count;
        }
    }
    if(*leader == my_rank)
    {
        map<int,int> rank_map;
        for(int i = 0; i < size; ++i)
        {
            if(nccl_ranks[i] == 0)
            {
                rank_map[hosts[i]] = intercomm_ranks[i];
            }
            else
            {
                intercomm_ranks[i] = rank_map[hosts[i]];
            }
        }
    }
    /*
    printf("Rank %d: hosts\n", my_rank);
    for(int i = 0; i < size; ++i)
    {
        printf("Rank %d: %d: %d\n", my_rank, i, hosts[i]);
    }
    printf("Rank %d: nccl_ranks\n", my_rank);
    for(int i = 0; i < size; ++i)
    {
        printf("Rank %d: %d: %d\n", my_rank, i, nccl_ranks[i]);
    }
    printf("Rank %d: intercomm_ranks\n", my_rank);
    for(int i = 0; i < size; ++i)
    {
        printf("Rank %d: %d: %d\n", my_rank, i, intercomm_ranks[i]);
    }
    */
}
