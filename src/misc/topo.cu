/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "topo.h"
#include "nvmlwrap.h"
#include "nccl.h"
#include <ctype.h>

#define CHECK(cmd) do { \
  int ret = cmd; \
  if (ret != ncclSuccess) \
    return 0; \
} while (0);

#define CUDA_CHECK(cmd) do { \
  int ret = cmd; \
  if (ret != cudaSuccess) { \
    return 0; \
  } \
} while (0);

int getTopoMatrix(int* matrix, int* devs, int ndev) {
  int minLinks = ndev;
  char busIds[ndev][NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
  for(int g=0; g<ndev; ++g) {
    // Get bus IDs to get around CUDART re-indexing.
    CUDA_CHECK(cudaDeviceGetPCIBusId(busIds[g], NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE, devs[g]));
    // Contrary to CUDA, NVML uses lower case
    for (int c=0; c<NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE; c++) {
      if (busIds[g][c] == 0) break;
      busIds[g][c] = tolower(busIds[g][c]);
    }
  }

  for(int r1=0; r1<ndev; ++r1) {
    int links = 0;
    nvmlDevice_t nvmlDev;
    CHECK(wrapNvmlDeviceGetHandleByPciBusId(busIds[r1], &nvmlDev));

    // Check whom we are connected to through NVLink on this device

    for(int l=0; l<NVML_NVLINK_MAX_LINKS; ++l) {
      // nvmlDeviceGetNvLinkState() reports whether a link is enabled or not.
      // In practice, it works only on PASCAL (tested GP100-SXM2, GP100-PCIE,
      // and GP102).
      nvmlEnableState_t linkState;
      CHECK(wrapNvmlDeviceGetNvLinkState(nvmlDev, l, &linkState));
      if (linkState == NVML_FEATURE_DISABLED) continue;

      // nvmlDeviceGetNvLinkCapability(NVML_NVLINK_CAP_P2P_SUPPORTED) would seem to
      // report whether the NVLink connects to a peer GPU (versus a POWER CPU?). I
      // don't know whether nvmlDeviceGetNvLinkRemotePciInfo() would succeed in
      // the POWER CPU case, so it seems best to check this as well.
      unsigned canP2P;
      CHECK(wrapNvmlDeviceGetNvLinkCapability(nvmlDev, l, NVML_NVLINK_CAP_P2P_SUPPORTED, &canP2P));
      if (!canP2P) continue;

      // nvmlDeviceGetNvLinkRemotePciInfo() will return NVML_ERROR_NOT_SUPPORTED
      // if the links don't exist, or are disabled. So checking for that return
      // here would probably make the nvmlDeviceGetNvLinkState check above
      // redundant. Presumably, we still need to check the P2P capability above,
      // since even non-GPUs would posses PCI info.
      nvmlPciInfo_t remoteProc;
      CHECK(wrapNvmlDeviceGetNvLinkRemotePciInfo(nvmlDev, l, &remoteProc));

      for (int r2=0; r2<ndev; ++r2) {
        if (strncmp(busIds[r2], remoteProc.busId, NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE) == 0) {
          matrix[r1*ndev+r2]++;
          links++;
          break;
        }
      }
    }
    if (links == 0) {
      // One device is not connected to any other. In that case, there is no
      // point in building rings.
      return 0;
    }
    minLinks = min(minLinks, links);
  }
  return minLinks;
}

static int computeRingsRec(int* matrix, int n, int *rings, int currentRing, int nRingsMax, int* inTheRing, int current, int remaining) {
  int nrings = 0;
  int* line = matrix+current*n;
  inTheRing[current] = 1;
  rings[currentRing*n+n-remaining-1] = current;
  if (remaining == 0) {
    if (line[0] > 0) {
      if (currentRing+1 == nRingsMax) {
        nrings = 1;
      } else {
	line[0]--;
	for (int i=0; i<n; i++) inTheRing[i] = 0;
	rings[(currentRing+1)*n] = 0;
	nrings = 1 + computeRingsRec(matrix, n, rings, currentRing+1, nRingsMax, inTheRing, 0, n-1);
	line[0]++;
	for (int i=0; i<n; i++) inTheRing[i] = 1;
      }
    }
  } else {
    int rings_save[nRingsMax*n];
    int offset = currentRing*n+n-remaining;
    for (int i=1; i<n; i++) {
      if (inTheRing[i] == 0 && line[i] > 0) {
        line[i]--;
        int nr = computeRingsRec(matrix, n, rings, currentRing, nRingsMax, inTheRing, i, remaining-1);
        if (nr > nrings) {
          nrings = nr;
          rings_save[offset] = i;
          // Save the rest of the rings
          for (int r=offset+1; r<(nrings+currentRing)*n; r++) {
            rings_save[r] = rings[r];
          }
          if (nrings + currentRing == nRingsMax) {
            // We found an optimal solution. Let's stop there.
            break;
          }
        }
        line[i]++;
      }
    }
    for (int r=offset; r<(nrings+currentRing)*n; r++) {
      rings[r] = rings_save[r];
    }
  }
  inTheRing[current] = 0;
  return nrings;
}

int computeRings(int* matrix, int n, int* rings, int nRingsMax) {
  int* inTheRing = (int*)malloc(sizeof(int)*n);
  for (int i=0; i<n; i++) inTheRing[i] = 0;
  rings[0] = 0;
  int nrings = computeRingsRec(matrix, n, rings, 0, nRingsMax, inTheRing, 0, n-1);
  free(inTheRing);
  return nrings;
}

void ncclTopoGetRings(int* devs, int n, int* nringsPtr, int** ringsPtr) {
  int* matrix = (int*)malloc(sizeof(int)*n*n);
  for (int i=0; i<n*n; i++)
    matrix[i] = 0;

  int nRings = 0;
  int nRingsMax = getTopoMatrix(matrix, devs, n);
  int* tmpRings = (int*)malloc(sizeof(int)*n*nRingsMax);

  if (nRingsMax == 0) {
    nRings = 0;
    goto end;
  }
  nRings = computeRings(matrix, n, tmpRings, nRingsMax);
  if (nRings < nRingsMax && n < 8) {
    // Try with 2x oversubscription
    for (int x=0; x<n; x++)
      for (int y=0; y<n; y++)
        matrix[x*n+y] = matrix[x*n+y]*2;
    int* tmpRings2 = (int*)malloc(sizeof(int)*n*nRingsMax*2);
    int nRings2 = computeRings(matrix, n, tmpRings2, nRingsMax*2);

    if (nRings2 > nRings*2) {
      free(tmpRings);
      nRings = nRings2;
      tmpRings = tmpRings2;
    } else {
      free(tmpRings2);
    }
  }

end:
  free(matrix);
  if (nRings > 0) {
    *ringsPtr = (int*)malloc(sizeof(int)*n*nRings);
    memcpy(*ringsPtr, tmpRings, sizeof(int)*n*nRings);
  }
  free(tmpRings);
  *nringsPtr = nRings;
}

