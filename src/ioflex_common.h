#ifndef __IOFLEX_COMMON
#define __IOFLEX_COMMON

#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <float.h>
#include <limits.h>
#include <errno.h>
#include <mpi.h>

#define FORWARD_DECL(name, ret, args) \
    ret(*__act_##name) args = NULL;

#define DECL(__name) __name

#define IOFLEX_WRAPPER_MAP(__func, __ret, __args, __fcall) \
    __ret __func __args __attribute__((alias(#__fcall)));

#define MAP_OR_FAIL(func)                                                \
    if (!(__act_##func))                                                 \
    {                                                                    \
        __act_##func = dlsym(RTLD_NEXT, #func);                          \
        if (!(__act_##func))                                             \
        {                                                                \
            fprintf(stderr, "IOFlex failed to map symbol: %s\n", #func); \
            exit(1);                                                     \
        }                                                                \
    }

typedef struct mpi_configs
{
    char romio_ds_read[MPI_MAX_INFO_VAL];
    char romio_ds_write[MPI_MAX_INFO_VAL];
    char romio_cb_read[MPI_MAX_INFO_VAL];
    char romio_cb_write[MPI_MAX_INFO_VAL];
    char romio_filesystem_type[MPI_MAX_INFO_VAL];
    char cb_config_list[MPI_MAX_INFO_VAL];
    char striping_unit[MPI_MAX_INFO_VAL];
    char striping_factor[MPI_MAX_INFO_VAL];
    char cb_nodes[MPI_MAX_INFO_VAL];
    char cb_buffer_size[MPI_MAX_INFO_VAL];
} mpi_configs;

void set_mpi_info_config(MPI_Info info);

#endif
