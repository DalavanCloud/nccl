/*
 * Copyright (c) 2014      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2014-2015 NVIDIA Corporation.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"

#include <string.h>

#include "mpi.h"
#include "ompi/constants.h"
#include "coll_nccl.h"

#include "coll_nccl_debug.h"

int mca_coll_nccl_output = 0;

/*
 * Public string showing the coll ompi_cuda component version number
 */
const char *mca_coll_nccl_component_version_string =
    "Open MPI NCCL collective MCA component version " OMPI_VERSION;

/*
 * Local function
 */
static int nccl_register(void);
static int nccl_open(void);
static int nccl_close(void);

/*
 * Instantiate the public struct with all of our public information
 * and pointers to our public functions in it
 */

mca_coll_nccl_component_t mca_coll_nccl_component = {
    {
        /* First, the mca_component_t struct containing meta information
         * about the component itself */

        {
            MCA_COLL_BASE_VERSION_2_0_0,

            /* Component name and version */
            "nccl",
            OMPI_MAJOR_VERSION,
            OMPI_MINOR_VERSION,
            OMPI_RELEASE_VERSION,

            /* Component open and close functions */
            nccl_open,
            nccl_close,
            NULL,
            nccl_register
        },
        {
            /* The component is checkpoint ready */
            MCA_BASE_METADATA_PARAM_CHECKPOINT
        },

        /* Initialization / querying functions */

        mca_coll_nccl_init_query,
        mca_coll_nccl_comm_query
    },

    /* nccl-specific component information */

    /* Priority: since we are the most specific component, make it above all point to point collectives including self and all collective components*/
    .priority = 80,
    /* Treshold: messages smaller than the treshild will not be sent using NCCL */
    .treshold = 51200,
    /* Verbosity level */
    .verbose = 0,
    /*Segment size, 1MB by default*/
    .segment_size = 1 << 20,
    /*Pipeline segment size, 4 MB by default*/
    .pipeline_segment_size = 4 * 1024*1024
};


static int nccl_register(void)
{
    (void) mca_base_component_var_register(&mca_coll_nccl_component.super.collm_version,
                                           "priority", "Priority of the nccl coll component; only relevant if barrier_before or barrier_after is > 0",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_6,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &mca_coll_nccl_component.priority);
    mca_coll_nccl_component.treshold = 51200;
    (void) mca_base_component_var_register(&mca_coll_nccl_component.super.collm_version,
                                           "treshold", "Treshold for using NCCL. Messages smaller than the treshold will not be using NCCL",
                                           MCA_BASE_VAR_TYPE_SIZE_T, NULL, 0, 0,
                                           OPAL_INFO_LVL_6,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &mca_coll_nccl_component.treshold);
    mca_coll_nccl_component.verbose = 0;
    (void) mca_base_component_var_register(&mca_coll_nccl_component.super.collm_version,
                                           "verbose", "Level of verbosity for NCCL coll component",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_6,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &mca_coll_nccl_component.verbose);
    mca_coll_nccl_component.segment_size = 1 << 20;
    (void) mca_base_component_var_register(&mca_coll_nccl_component.super.collm_version,
                                           "segment_size", "Segment size for multi node communication. 1 MB by default",
                                           MCA_BASE_VAR_TYPE_SIZE_T, NULL, 0, 0,
                                           OPAL_INFO_LVL_6,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &mca_coll_nccl_component.segment_size);
    mca_coll_nccl_component.pipeline_segment_size = 4 * 1024 * 1024;
    (void) mca_base_component_var_register(&mca_coll_nccl_component.super.collm_version,
                                           "pipeline_segment_size", "Segment size for pipelining multi node communication with NCCL. 4 MB by default. The best performance achieved when it is a multiple of the number of GPUs in a node",
                                           MCA_BASE_VAR_TYPE_SIZE_T, NULL, 0, 0,
                                           OPAL_INFO_LVL_6,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &mca_coll_nccl_component.pipeline_segment_size);
    return OMPI_SUCCESS;
}

static int nccl_open(void)
{
    mca_coll_nccl_output = opal_output_open(NULL);
    opal_output_set_verbosity(mca_coll_nccl_output, mca_coll_nccl_component.verbose);
    return OMPI_SUCCESS;
}

static int nccl_close(void)
{
    opal_output_close(mca_coll_nccl_output);
    return OMPI_SUCCESS;
}
