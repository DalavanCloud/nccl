/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"

static void recFillMyGroup(int rank, int* groups, int nranks, int* matrix, int transport) {
  groups[rank] = -2;
  for (int r=0; r<nranks; r++) {
    if (groups[r] == -2) continue;
    if (matrix[rank*nranks+r] <= transport) recFillMyGroup(r, groups, nranks, matrix, transport);
  }
}

static void fillMyGroup(int rank, int* groups, int nranks, int* matrix, int transport) {
  for (int r=0; r<nranks; r++) groups[r] = -1;
  recFillMyGroup(rank, groups, nranks, matrix, transport);
}

static void recNumberMyGroup(int rank, int group, int* groups, int nranks, int* matrix, int transport) {
  groups[rank] = group;
  for (int r=0; r<nranks; r++) {
    if (groups[r] == -1 || groups[r] >= 0) continue;
    if (matrix[rank*nranks+r] < transport)
      recNumberMyGroup(r, group, groups, nranks, matrix, transport);
  }
}

static int numberMyGroup(int* groups, int nranks, int* matrix, int transport) {
  int group = 0;
  for (int i=0; i<nranks; i++) {
    if (groups[i] == -2)
      recNumberMyGroup(i, group++, groups, nranks, matrix, transport);
  }
  return group;
}

static int fillGroups(int rank, int* groups, int nranks, int* matrix, int transport) {
  fillMyGroup(rank, groups, nranks, matrix, transport);
  return numberMyGroup(groups, nranks, matrix, transport);
}

ncclResult_t ncclGetRings(int* nrings, int rank, int nranks, int* transports, int* values, int* prev, int* next) {
  *nrings = 0;
  if (nranks == 1) return ncclSuccess;

  int pattern = 0;
  int nringsTmp;
  int prevTmp[nranks*MAXRINGS];
  int nextTmp[nranks*MAXRINGS];
  do {
    for (int i=0; i<nranks*MAXRINGS; i++) prevTmp[i] = nextTmp[i] = -1;
    nringsTmp = MAXRINGS;
    for (int t=NTRANSPORTS-1; t>=0; t--) {
      int groups[nranks];
      int ngroups = fillGroups(rank, groups, nranks, transports, t);
      if (ngroups > 1) {
        /* Reduce the scope to the local ranks and sort them by group */
        int idxToRank[nranks];
        int rankToIdx[nranks];
        for (int i=0; i<nranks; i++) idxToRank[i] = rankToIdx[i] = -1;
        int nidx = 0;
        for (int g=0; g<ngroups; g++) {
          for (int r=0; r<nranks; r++) {
            if (groups[r] == g) {
              rankToIdx[r] = nidx;
              idxToRank[nidx] = r;
              nidx++;
            }
          }
        }
        /* Extract groups */
        int subgroups[nidx];
        for (int i=0; i<nidx; i++) {
          subgroups[i] = groups[idxToRank[i]];
        }
        /* Extract values */
        int subvalues[nidx*nidx];
        for (int i=0; i<nidx; i++) {
          for (int j=0; j<nidx; j++) {
            if (transports[idxToRank[i]*nranks+idxToRank[j]] == t)
              subvalues[i*nidx+j] = values[idxToRank[i]*nranks+idxToRank[j]];
            else
              subvalues[i*nidx+j] = 0;
          }
        }
        /* Extract prev/next */
        int subprev[nidx*nringsTmp];
        int subnext[nidx*nringsTmp];
        for (int i=0; i<nidx*nringsTmp; i++) {
          subprev[i] = subnext[i] = -1;
        }
        for (int r=0; r<nringsTmp; r++) {
          int start = -1, end = -1;
          for (int i=0; i<nranks; i++) {
            if (rankToIdx[i] == -1) continue;
            if (prevTmp[r*nranks+i] != -1) start = i;
            if (nextTmp[r*nranks+i] != -1) end = i;
          }
          if (start != -1 && end != -1) {
            subprev[r*nidx+rankToIdx[start]] = rankToIdx[end];
            subnext[r*nidx+rankToIdx[end]] = rankToIdx[start];
          }
        }
        /* Get rings */
        NCCLCHECK(ncclTransports[t].getRings(nidx, ngroups, subgroups, subvalues, &nringsTmp, subprev, subnext, pattern));
        /* Merge prev/next */
        for (int r=0; r<nringsTmp; r++) {
          for (int i=0; i<nidx; i++) {
            if ((prevTmp[r*nranks+idxToRank[i]] == -1) && (subprev[r*nidx+i] != -1)) prevTmp[r*nranks+idxToRank[i]] = idxToRank[subprev[r*nidx+i]];
            if ((nextTmp[r*nranks+idxToRank[i]] == -1) && (subnext[r*nidx+i] != -1)) nextTmp[r*nranks+idxToRank[i]] = idxToRank[subnext[r*nidx+i]];
          }
        }
      }
    }
    pattern++;
    if (nringsTmp > *nrings) {
      *nrings = nringsTmp;
      for (int i=0; i<nranks*(*nrings); i++) {
        prev[i] = prevTmp[i];
        next[i] = nextTmp[i];
      }
    }
  } while (nringsTmp != 0);

  return ncclSuccess;
}

