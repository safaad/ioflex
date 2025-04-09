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

typedef enum
{
    ROMIO_DS_READ,
    ROMIO_DS_WRITE,
    ROMIO_CB_READ,
    ROMIO_CB_WRITE,
    ROMIO_FS_TYPE,
    CB_CONFIG_LIST,
    STRIPING_UNIT,
    STRIPING_FACTOR,
    CB_NODES,
    CB_BUFFER_SIZE,
    ROMIO_NO_INDEP_RW,
    IND_RD_BUFFER_SIZE,
    IND_WR_BUFFER_SIZE,
    CONFIG_FIELD_COUNT

} mpi_config_field;

extern char mpi_configs[CONFIG_FIELD_COUNT][MPI_MAX_INFO_VAL];

void set_mpi_info_config(MPI_Info info);

#endif
