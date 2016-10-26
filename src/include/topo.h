/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef TOPO_H_
#define TOPO_H_

void ncclTopoGetRings(int* devs, int n, int* nringsPtr, int** ringsPtr);

#endif
