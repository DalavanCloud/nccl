/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"

static void recFillCoords(int rank, int nranks, int* matrix, int* current, int* coords, int* idx, int* rankToIdx, int* idxToRank) {
  for (int t=0; t<NTRANSPORTS; t++) {
    coords[rank*NTRANSPORTS+t] = current[t];
  }
  rankToIdx[rank] = *idx;
  idxToRank[*idx] = rank;
  for (int t=0; t<NTRANSPORTS; t++) {
    for (int r=0; r<nranks; r++) {
      if (coords[r*NTRANSPORTS] != -1) continue;
      if (matrix[rank*nranks+r] == t) {
        current[t]++;
        (*idx)++;
        recFillCoords(r, nranks, matrix, current, coords, idx, rankToIdx, idxToRank);
      }
    }
    current[t] = 0;
  }
}

static void fillCoords(int nranks, int* matrix, int* coords, int* rankToIdx, int* idxToRank) {
  // Set 0's coordinates to (0,0, ... 0)
  int current[NTRANSPORTS];
  for (int i=0; i<NTRANSPORTS; i++) current[i] = 0;
  int index = 0;
  recFillCoords(0, nranks, matrix, current, coords, &index, rankToIdx, idxToRank);
}

ncclResult_t ncclGetRings(int* nrings, int rank, int nranks, int* transports, int* values, int* prev, int* next) {
  *nrings = 0;
  if (nranks == 1) return ncclSuccess;

  // Compute hierarchical topology groups, indexes, and rank<->index tables
  int coords[nranks*NTRANSPORTS];
  int globalIdxToRank[nranks];
  int globalRankToIdx[nranks];
  for (int i=0; i<nranks*NTRANSPORTS; i++) coords[i] = -1;
  fillCoords(nranks, transports, coords, globalRankToIdx, globalIdxToRank);

  int minScore = NCCL_MAX_SCORE;
  int nringsTmp;
  int prevTmp[nranks*MAXRINGS];
  int nextTmp[nranks*MAXRINGS];
  do {
    for (int i=0; i<nranks*MAXRINGS; i++) prevTmp[i] = nextTmp[i] = -1;
    nringsTmp = MAXRINGS;
    for (int t=NTRANSPORTS-1; t>=0; t--) {
      int idxToRank[nranks];
      int rankToIdx[nranks];
      int groups[nranks];
      int subgroups[nranks];
      for (int i=0; i<nranks; i++) idxToRank[i] = rankToIdx[i] = -1;
      
      int nidx = 0;
      for (int i=0; i<nranks; i++) {
        // Extract only ranks in the same local area as rank
        // We need to extract them in the topological order, hence we iterate over indexes, not ranks
        int r = globalIdxToRank[i];
        int sameLocal = 1;
        for (int tr = NTRANSPORTS-1; tr > t; tr--) if (coords[r*NTRANSPORTS+tr] != coords[rank*NTRANSPORTS+tr]) sameLocal = 0;
        if (!sameLocal) continue;

        groups[nidx] = coords[r*NTRANSPORTS+t];
        subgroups[nidx] = t ? coords[r*NTRANSPORTS+t-1] : nidx;
        rankToIdx[r] = nidx;
        idxToRank[nidx] = r;
        nidx++;
      }
      // Coords should be ordered
      int ngroups = groups[nidx-1] + 1;

      if (ngroups > 1) {
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
        NCCLCHECK(ncclTransports[t].getRings(nidx, groups, subgroups, subvalues, &nringsTmp, subprev, subnext, minScore));
        /* Merge prev/next */
        for (int r=0; r<nringsTmp; r++) {
          for (int i=0; i<nidx; i++) {
            if ((prevTmp[r*nranks+idxToRank[i]] == -1) && (subprev[r*nidx+i] != -1)) prevTmp[r*nranks+idxToRank[i]] = idxToRank[subprev[r*nidx+i]];
            if ((nextTmp[r*nranks+idxToRank[i]] == -1) && (subnext[r*nidx+i] != -1)) nextTmp[r*nranks+idxToRank[i]] = idxToRank[subnext[r*nidx+i]];
          }
        }
        for (int r=0; r<nringsTmp; r++) {
        //printf("[%d] [%d] [%d] [%d] Prev ", rank, minScore, t, r); for (int i=0; i<nranks; i++) printf("%d ", prevTmp[r*nranks+i]); printf("\n");
        //printf("[%d] [%d] [%d] [%d] Next ", rank, minScore, t, r); for (int i=0; i<nranks; i++) printf("%d ", nextTmp[r*nranks+i]); printf("\n");
        }
      }
    }
    minScore--;
    if (nringsTmp > *nrings) {
      *nrings = nringsTmp;
      for (int i=0; i<nranks*(*nrings); i++) {
        prev[i] = prevTmp[i];
        next[i] = nextTmp[i];
      }
    }
  } while (nringsTmp == 0 && minScore);

  if (*nrings == 0) {
    WARN("Could not create rings");
    return ncclInternalError;
  }
  return ncclSuccess;
}

