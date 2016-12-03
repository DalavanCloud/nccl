/*************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "debug.h"
#include <stdint.h>
#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <errno.h>
#include <malloc.h>
#include <sys/ioctl.h>
#include <asm/types.h>

/*===== gdrdrv.h Kernel API =====*/

#define GDRDRV_IOCTL                 0xDA

typedef __u32 gdr_hnd_t;
#define GDR_HANDLE_MASK ((1UL<<32)-1)

//-----------

struct GDRDRV_IOC_PIN_BUFFER_PARAMS
{
    // in
    __u64 addr;
    __u64 size;
    __u64 p2p_token;
    __u32 va_space;
    // out
    gdr_hnd_t handle;
};

#define GDRDRV_IOC_PIN_BUFFER _IOWR(GDRDRV_IOCTL, 1, struct GDRDRV_IOC_PIN_BUFFER_PARAMS)

//-----------

struct GDRDRV_IOC_UNPIN_BUFFER_PARAMS
{
    // in
    gdr_hnd_t handle;
};

#define GDRDRV_IOC_UNPIN_BUFFER _IOWR(GDRDRV_IOCTL, 2, struct GDRDRV_IOC_UNPIN_BUFFER_PARAMS *)

//-----------

struct GDRDRV_IOC_GET_CB_FLAG_PARAMS
{
    // in
    gdr_hnd_t handle;
    // out
    __u32 flag;
};

#define GDRDRV_IOC_GET_CB_FLAG _IOWR(GDRDRV_IOCTL, 3, struct GDRDRV_IOC_GET_CB_FLAG_PARAMS *)

//-----------

struct GDRDRV_IOC_GET_INFO_PARAMS
{
    // in
    gdr_hnd_t handle;
    // out
    __u64 va;
    __u64 mapped_size;
    __u32 page_size;
    __u32 tsc_khz;
    __u64 tm_cycles;
};

#define GDRDRV_IOC_GET_INFO _IOWR(GDRDRV_IOCTL, 4, struct GDRDRV_IOC_GET_INFO_PARAMS *)

//-----------

struct GDRDRV_IOC_PIN_BUFFER_EXT_PARAMS
{
    // in
    __u64 addr;
    __u64 size;
    __u64 p2p_token;
    __u32 va_space;
    // in
    __u32 npages;
    // out
    __u32 iscontig;
    __u64 paddr;
    __u64 *paddrs;
    gdr_hnd_t handle;
};

#define GDRDRV_IOC_PIN_BUFFER_EXT _IOWR(GDRDRV_IOCTL, 5, struct GDRDRV_IOC_PIN_BUFFER_EXT_PARAMS)

struct GDRDRV_IOC_PREPARE_HANDLE_PHYSICAL_EXT_PARAMS
{
    // in
    __u32 npages;
    __u64 pagesize;
    __u64 *paddrs;
    // out
    gdr_hnd_t handle;
};

#define GDRDRV_IOC_PREPARE_HANDLE_PHYSICAL_EXT _IOWR(GDRDRV_IOCTL, 6, struct GDRDRV_IOC_PREPARE_HANDLE_PHYSICAL_EXT_PARAMS)

struct GDRDRV_IOC_DESTROY_HANDLE_EXT_PARAMS
{
    // in
    gdr_hnd_t handle;
};

#define GDRDRV_IOC_DESTROY_HANDLE_EXT _IOWR(GDRDRV_IOCTL, 7, struct GDRDRV_IOC_DESTROY_HANDLE_EXT_PARAMS)
/*===== End gdrdrv.h Kernel API =====*/

#define GPU_PAGE_SHIFT   16
#define GPU_PAGE_SIZE    ((unsigned long)1 << GPU_PAGE_SHIFT)
#define GPU_PAGE_OFFSET  (GPU_PAGE_SIZE-1)
#define GPU_PAGE_MASK    (~GPU_PAGE_OFFSET)

#define CPU_PAGE_SHIFT   12
#define CPU_PAGE_SIZE    ((unsigned long)1 << CPU_PAGE_SHIFT)
#define CPU_PAGE_OFFSET  (CPU_PAGE_SIZE-1)
#define CPU_PAGE_MASK    (~CPU_PAGE_OFFSET)

struct gdr {
    int fd;
};

typedef struct gdr *gdr_t;
typedef uint32_t gdr_mh_t;

// After pinning, info struct contains details of the mapped area.  
//
// Note that both info->va and info->mapped_size might be different from
// the original address passed to gdr_pin_buffer due to aligning happening
// in the kernel-mode driver
struct gdr_info {
    uint64_t va;
    uint64_t mapped_size;
    uint32_t page_size;
    uint64_t tm_cycles;
    uint32_t cycles_per_ms;
};
typedef struct gdr_info gdr_info_t;

// Map device memory buffer on GPU BAR1, returning an handle.
// Memory is still not accessible to user-space.
typedef struct {
    gdr_mh_t handle; 
    uint32_t pagesize;
    uint32_t npages;
    uint32_t iscontig;
    uint64_t paddr;
    uint64_t *paddrs;
} gdr_mh_ext_t;

gdr_t gdr_open()
{
    gdr_t g = NULL;
    const char *gdrinode = "/dev/gdrdrv";

    g = (gdr_t)calloc(1, sizeof(*g));
    if (!g) {
        WARN("GDCOPY : error while allocating memory\n");
        return NULL;
    }

    int fd = open(gdrinode, O_RDWR);
    if (-1 == fd ) {
        int ret = errno;
        INFO("GDCOPY : error opening driver (errno=%d/%s)\n", ret, strerror(ret));
        free(g);
        return NULL;
    }

    g->fd = fd;

    return g;
}

int gdr_close(gdr_t g)
{
    int ret = 0;
    int retcode = close(g->fd);
    if (-1 == retcode) {
        ret = errno;
        WARN("GDCOPY : error closing driver (errno=%d/%s)\n", ret, strerror(ret));
    }
    g->fd = 0;
    free(g);
    return ret;
}

int gdr_pin_buffer(gdr_t g, unsigned long addr, size_t size, uint64_t p2p_token, uint32_t va_space, gdr_mh_t *handle)
{
    int ret = 0;
    int retcode;

    struct GDRDRV_IOC_PIN_BUFFER_PARAMS params;
    params.addr = addr;
    params.size = size;
    params.p2p_token = p2p_token;
    params.va_space = va_space;
    params.handle = 0;

    retcode = ioctl(g->fd, GDRDRV_IOC_PIN_BUFFER, &params);
    if (0 != retcode) {
        ret = errno;
        WARN("GDCOPY : ioctl error (errno=%d)\n", ret);
    }
    *handle = params.handle;

    return ret;
}

int gdr_pin_buffer_ext(gdr_t g, unsigned long addr, size_t size, uint64_t p2p_token, uint32_t va_space, gdr_mh_ext_t *handle)
{
    int ret = 0;
    int retcode, npages;
    unsigned long virt_start, virt_end, adjusted_size;

    struct GDRDRV_IOC_PIN_BUFFER_EXT_PARAMS params;
    params.addr = addr;
    params.size = size;
    params.p2p_token = p2p_token;
    params.va_space = va_space;
    params.handle = 0;

    virt_start = addr & GPU_PAGE_MASK;
    virt_end = addr + size - 1;
    adjusted_size = virt_end - virt_start + 1;
    npages = (adjusted_size >> GPU_PAGE_SHIFT) + !!(adjusted_size & GPU_PAGE_OFFSET);	
    params.npages = npages;

    //printf("adjusted_size: %d page count: %d virt_start: %p addr: %p\n", adjusted_size, params.npages, virt_start, addr);

    retcode = ioctl(g->fd, GDRDRV_IOC_PIN_BUFFER_EXT, &params);
    if (0 != retcode) {
        ret = errno;
        WARN("GDCOPY : ioctl error (errno=%d)\n", ret);
        goto out;
    }

    handle->pagesize = GPU_PAGE_SIZE;
    handle->npages = params.npages;
    handle->paddr = params.paddr;
    handle->iscontig = params.iscontig;
    handle->handle = params.handle;

out:
    return ret;
}

int gdr_unpin_buffer(gdr_t g, gdr_mh_t handle)
{
    int ret = 0;
    int retcode;

    struct GDRDRV_IOC_UNPIN_BUFFER_PARAMS params;
    params.handle = handle;

    retcode = ioctl(g->fd, GDRDRV_IOC_UNPIN_BUFFER, &params);
    if (0 != retcode) {
        ret = errno;
        WARN("GDCOPY : ioctl error (errno=%d)\n", ret);
    }

    return ret;
}

int gdr_unpin_buffer_ext(gdr_t g, gdr_mh_ext_t handle)
{
    int ret = 0;
    int retcode;

    struct GDRDRV_IOC_UNPIN_BUFFER_PARAMS params;
    params.handle = handle.handle;

    retcode = ioctl(g->fd, GDRDRV_IOC_UNPIN_BUFFER, &params);
    if (0 != retcode) {
        ret = errno;
        WARN("GDCOPY : ioctl error (errno=%d)\n", ret);
    }

    return ret;
}

int gdr_get_callback_flag(gdr_t g, gdr_mh_t handle, int *flag)
{
    int ret = 0;
    int retcode;

    struct GDRDRV_IOC_GET_CB_FLAG_PARAMS params;
    params.handle = handle;

    retcode = ioctl(g->fd, GDRDRV_IOC_GET_CB_FLAG, &params);
    if (0 != retcode) {
        ret = errno;
        WARN("GDCOPY : ioctl error (errno=%d)\n", ret);
    } else
        *flag = params.flag;

    return ret;
}

int gdr_get_info(gdr_t g, gdr_mh_t handle, gdr_info_t *info)
{
    int ret = 0;
    int retcode;

    struct GDRDRV_IOC_GET_INFO_PARAMS params;
    params.handle = handle;

    retcode = ioctl(g->fd, GDRDRV_IOC_GET_INFO, &params);
    if (0 != retcode) {
        ret = errno;
        WARN("GDCOPY : ioctl error (errno=%d)\n", ret);
    } else {
        info->va          = params.va;
        info->mapped_size = params.mapped_size;
        info->page_size   = params.page_size;
        info->tm_cycles   = params.tm_cycles;
        info->cycles_per_ms = params.tsc_khz;
    }
    return ret;
}

int gdr_map(gdr_t g, gdr_mh_t handle, void **ptr_va, size_t size)
{
    int ret = 0;
    gdr_info_t info = {0,};

    ret = gdr_get_info(g, handle, &info);
    if (ret) {
	WARN("GDCOPY : error getting info \n");
        return ret;
    }
    size_t rounded_size = (size + CPU_PAGE_SIZE - 1) & CPU_PAGE_MASK;
    off_t magic_off = (off_t)handle << CPU_PAGE_SHIFT;
    void *mmio;

    mmio = mmap(NULL, rounded_size, PROT_READ|PROT_WRITE, MAP_SHARED, g->fd, magic_off);
    if (mmio == MAP_FAILED) {
        int __errno = errno;
        mmio = NULL;
        WARN("GDCOPY : can't mmap BAR, error=%s(%d) rounded_size=%zu offset=%llx handle=%x\n",
                strerror(__errno), __errno, rounded_size, (long long unsigned)magic_off, handle);
        ret = __errno;
    }

    *ptr_va = mmio;

    return ret;
}

int gdr_unmap(gdr_t g, gdr_mh_t handle, void *va, size_t size)
{
    int ret = 0;
    int retcode = 0;

    size_t rounded_size = (size + CPU_PAGE_SIZE - 1) & CPU_PAGE_MASK;

    retcode = munmap(va, rounded_size);
    if (-1 == retcode) {
        int __errno = errno;
        WARN("GDCOPY : can't unmap BAR, error=%s(%d) rounded_size=%zu\n",
                strerror(__errno), __errno, rounded_size);
        ret = __errno;
    }

    return ret;
}

/* NCCL specific function */

static gdr_t gd_handle;
static int gd_initialized = 0;

void* gdptr(void* devptr, int size) {
  if (gd_initialized == 0) {
    char* str = getenv("NCCL_GDCOPY_DISABLE");
    int gdr_disable = str ? atoi(str) : 0;
    if (gdr_disable == 1) {
      gd_handle = NULL;
    } else {
      gd_handle = gdr_open();
    }
    gd_initialized = 1;
  }
  if (gd_handle == NULL) return NULL;
  gdr_mh_t memhandle;
  char* ptr;
  if (gdr_pin_buffer(gd_handle, (unsigned long)devptr, size, 0, 0, &memhandle) != 0) return NULL;

  gdr_info_t info;
  if (gdr_map(gd_handle, memhandle, (void**)(&ptr), size) != 0
      || gdr_get_info(gd_handle, memhandle, &info) != 0) {
    gdr_unpin_buffer(gd_handle, memhandle);
    return NULL;
  }
  int offset = info.va - (unsigned long)devptr;
  return ptr + offset;
}
