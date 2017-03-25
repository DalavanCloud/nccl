/*************************************************************************
 * Copyright (c) 2016-2017, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "net.h"

/* Parse user defined rings. Format is like :
 * "0 1|1 0|0 1 2 3|3 2 1 0|0 2 3 1|1 3 2 0|0 1 2 3 4 5 6 7|7 6 5 4 3 2 1 0"
 * Rings with a non-matching number of ranks are ignored so we can provide
 * rings for multiple cases.
 */
#define MAX_ENV_RANKS 512
static ncclResult_t parseRings(const char* str, int* nringsRet, int nranks, int* prev, int* next) {
  int ranks[MAX_ENV_RANKS];
  int nrings = 0;
  int rank = 0;
  int offset = 0;
  int status = 0; // 0 : between numbers, 1 : inside number
  do {
    int digit = str[offset] - '0';
    if (digit >= 0 && digit <= 9) {
      if (status == 0) {
        ranks[rank] = digit;
        status = 1;
      } else {
        ranks[rank] = ranks[rank]*10+digit;
      }
    } else {
      if (status == 1) {
        rank++;
        if (rank == MAX_ENV_RANKS) goto end;
      }
      status = 0;
      if (str[offset] == '|' || str[offset] == '\0') {
        int prevRank = ranks[rank-1];
        // Ignore rings if nranks doesn't match
        if (rank != nranks) goto newring;

        for (int r=0; r<nranks; r++) {
          int rank = ranks[r];
          // Ignore rings with ranks out of bounds
          if (rank < 0 || rank >= nranks) goto newring;
          // Ignore rings with duplicate ranks
          for (int i=0; i<r; i++)
            if (ranks[i] == rank) goto newring;

          next[nrings*nranks+prevRank] = rank;
          prev[nrings*nranks+rank] = prevRank;
          prevRank = rank;
        }
        nrings++;
newring:
        rank = 0;
      }
    }
  } while (str[offset++] != 0);
end:
  *nringsRet = nrings;
  return ncclSuccess;
}

/*
 * Ring creation algorithm
 * 
 * First, we establish hierarchical coordinates depending on the way ranks can 
 * communicate. After fillCoords, we have for each rank a unique 3-int array
 * {   node, pci_domain,   rank } corresponding to the three transports :
 * { 2[NET],     1[SHM], 0[P2P] }.
 * Also, we renumber ranks (to indexes) based on their growing coordinates.
 *
 * Then, we ask transports to connect groups together. We start with net, then
 * shm, then p2p. We maintain two arrays, prev and next, where values are equal
 * to -1 when ranks are not yet connected, and a rank otherwise. We never 
 * connect ranks outside our group, meaning that on 4 nodes of 2 sockets of 4 
 * ranks, if we are rank 13, we should see something like (provided we have a 
 * single net interface, hence a single ring) :
 *
 * Connecting all nodes                                <13>
 * 2[NET] : prev 31 -1 -1 -1 -1 -1 -1 -1  7 -1 -1 -1 -1 -1 -1 -1 15 -1 -1 -1 -1 -1 -1 -1 23 -1 -1 -1 -1 -1 -1 -1
 *          next -1 -1 -1 -1 -1 -1 -1  8 -1 -1 -1 -1 -1 -1 -1 16 -1 -1 -1 -1 -1 -1 -1 24 -1 -1 -1 -1 -1 -1 -1  0
 *
 * Connecting PCI domains (only inside the node)       <13>
 * 1[SHM] : prev 31 -1 -1 -1 -1 -1 -1 -1  7 -1 -1 -1 11 -1 -1 -1 15 -1 -1 -1 -1 -1 -1 -1 23 -1 -1 -1 -1 -1 -1 -1
 *          next -1 -1 -1 -1 -1 -1 -1  8 -1 -1 -1 12 -1 -1 -1 16 -1 -1 -1 -1 -1 -1 -1 24 -1 -1 -1 -1 -1 -1 -1  0
 *
 * Connecting ranks (only inside the PCI domain)       <13>
 * 0[P2P] : prev 31 -1 -1 -1 -1 -1 -1 -1  7 -1 -1 -1 11 12 13 14 15 -1 -1 -1 -1 -1 -1 -1 23 -1 -1 -1 -1 -1 -1 -1
 *          next -1 -1 -1 -1 -1 -1 -1  8 -1 -1 -1 12 13 14 15 16 -1 -1 -1 -1 -1 -1 -1 24 -1 -1 -1 -1 -1 -1 -1  0
 *
 * Hence, when we ask a transport to connect groups, we provide it with a subview of the ranks (except for net 
 * which always sees the full world). That way, P2P can bruteforce all combinations inside the node without
 * risking to explode in terms of combinations, and we scale better.
 *
 * Finally, we loop over Network scores to try to create rings with high scores (=locality) and decrease until
 * we get at least one ring.
 */

static int recIsConnected(int rank1, int rank2, int nranks, int* matrix, int transport, int* done) {
  if (matrix[rank1*nranks+rank2] == transport) return 1;
  done[rank1] = 1;
  for (int r=0; r<nranks; r++) {
    if (done[r] == 1) continue;
    if (matrix[rank1*nranks+r] == transport) {
      if (recIsConnected(r, rank2, nranks, matrix, transport, done)) return 1;
    }
  }
  return 0;
}

static int isConnected(int rank1, int rank2, int nranks, int* matrix, int transport) {
  int done[nranks];
  for (int r=0; r<nranks; r++) done[r] = 0;
  return recIsConnected(rank1, rank2, nranks, matrix, transport, done);
}

#define NEW_IDX(rank) do { \
  curRank = rank; \
  rankToIdx[curRank] = idx; \
  idxToRank[idx] = curRank; \
  for (int t=0; t<NTRANSPORTS; t++) coords[curRank*NTRANSPORTS+t] = current[t]; \
  transport = 0; \
  idx++; \
} while (0)

static ncclResult_t fillCoords(int nranks, int* matrix, int* coords, int* rankToIdx, int* idxToRank) {
  int current[NTRANSPORTS];
  for (int i=0; i<NTRANSPORTS; i++) current[i] = 0;
  int curRank, transport, idx = 0;
  NEW_IDX(0);
  while (transport < NTRANSPORTS) {
    for (int rank=0; rank<nranks; rank++) {
      if (coords[rank*NTRANSPORTS] != -1) continue;

      if (isConnected(curRank, rank, nranks, matrix, transport)) {
        current[transport]++;
        NEW_IDX(rank);// Resets transport = 0
        if (idx == nranks) return ncclSuccess;
      }
    }
    current[transport] = 0;
    transport++;
  }
  return ncclInternalError;
}

/* Users can force the number of threads with an environment variable */
ncclResult_t getEnvThreads(int* nthreads) {
  char* str = getenv("NCCL_NTHREADS");
  if (str && strlen(str) > 0) {
    int nt = atoi(str);
    if (nt != 128 && nt != 256 && nt != 512) {
      WARN("User-defined number of threads can only be 128, 256 or 512. Ignoring.");
    } else {
      *nthreads = nt;
    }
  }
  return ncclSuccess;
}

/* Main ring creation function */
ncclResult_t ncclGetRings(int* nrings, int* nthreads, int rank, int nranks, int* transports, int* values, int* prev, int* next) {
  *nrings = 0;

  if (nranks == 1) return ncclSuccess;

  char* str = getenv("NCCL_RINGS");
  if (str && strlen(str)>0) {
    int ret = parseRings(str, nrings, nranks, prev, next);
    if (ret == ncclSuccess && *nrings > 0) {
      if (rank == 0) INFO("%d ring(s) set by environment", *nrings);
      NCCLCHECK(getEnvThreads(nthreads));
      return ncclSuccess;
    }
    if (rank == 0) INFO("No valid ring found in environment, ignoring");
    *nrings = 0;
  }

  // Compute hierarchical topology groups, indexes, and rank<->index tables
  int coords[nranks*NTRANSPORTS];
  int globalIdxToRank[nranks];
  int globalRankToIdx[nranks];
  for (int i=0; i<nranks*NTRANSPORTS; i++) coords[i] = -1;
  NCCLCHECK(fillCoords(nranks, transports, coords, globalRankToIdx, globalIdxToRank));

  // Start with a high score, then decrease until we find rings
  int minScore = NCCL_MAX_SCORE;
  int nringsTmp;
  int prevTmp[nranks*MAXRINGS];
  int nextTmp[nranks*MAXRINGS];
  do {
    for (int i=0; i<nranks*MAXRINGS; i++) prevTmp[i] = nextTmp[i] = -1;
    nringsTmp = MAXRINGS;
    // Loop over transports to connect groups
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
 
      int ngroups = groups[nidx-1] + 1; // Coords should be ordered

      if (ngroups > 1) {
        /* Extract subvalues */
        int subvalues[nidx*nidx];
        for (int i=0; i<nidx; i++) {
          for (int j=0; j<nidx; j++) {
            if (transports[idxToRank[i]*nranks+idxToRank[j]] == t)
              subvalues[i*nidx+j] = values[idxToRank[i]*nranks+idxToRank[j]];
            else
              subvalues[i*nidx+j] = 0;
          }
        }
        /* Extract subprev/subnext */
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
        NCCLCHECK(ncclTransports[t].getRings(nidx, groups, subgroups, subvalues, &nringsTmp, subprev, subnext, minScore, nthreads));
        /* Merge subprev/subnext into prev/next */
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
    WARN("Could not create rings, falling back on simple ring");
    *nrings = 1;
    prev[rank] = (rank-1+nranks) % nranks;
    next[rank] = (rank+1)%nranks;
  }

  NCCLCHECK(getEnvThreads(nthreads));
  return ncclSuccess;
}

