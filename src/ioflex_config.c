#include <confuse.h>
#include "ioflex_common.h"
#include <limits.h>

bool MPI_CONFIG_IS_INITIALIZED = false;

char mpi_configs[CONFIG_FIELD_COUNT][MPI_MAX_INFO_VAL];

cfg_opt_t opts[] = {
    CFG_STR("romio_ds_read", 0, CFGF_NONE),
    CFG_STR("romio_ds_write", 0, CFGF_NONE),
    CFG_STR("romio_cb_read", 0, CFGF_NONE),
    CFG_STR("romio_cb_write", 0, CFGF_NONE),
    CFG_STR("romio_filesystem_type", 0, CFGF_NONE),
    CFG_STR("cb_config_list", 0, CFGF_NONE),
    CFG_STR("striping_unit", 0, CFGF_NONE),
    CFG_STR("striping_factor", 0, CFGF_NONE),
    CFG_STR("cb_nodes", 0, CFGF_NONE),
    CFG_STR("cb_buffer_size", 0, CFGF_NONE),
    CFG_STR("romio_no_indep_rw", 0, CFGF_NONE),
    CFG_STR("ind_rd_buffer_size", 0, CFGF_NONE),
    CFG_STR("ind_wr_buffer_size", 0, CFGF_NONE),
    CFG_END()};

void get_mpi_config()
{
    char filepath[PATH_MAX];
    const char *env = getenv("IOFLEX_HINTS");

    if (env != NULL)
    {
        strncpy(filepath, env, sizeof(filepath));
    }
    else
    {
        strncpy(filepath, "config.conf", sizeof(filepath));
    }

    cfg_t *cfg;
    cfg = cfg_init(opts, CFGF_NONE);
    if (cfg_parse(cfg, filepath) == CFG_PARSE_ERROR)
    {
        printf("Failed parsing config file or File not Found");
        exit(1);
    }
    size_t key = 0;
    cfg_opt_t opt;
    for (; (opt = opts[key]).type != CFGT_NONE; ++key)
    {
        if (cfg_getstr(cfg, opt.name))
        {
            strcpy(mpi_configs[key], cfg_getstr(cfg, opt.name));
        }
        else
        {
            memset(mpi_configs[key], 0, MPI_MAX_INFO_VAL);
        }
    }

    cfg_free(cfg);
}
__attribute__((constructor)) void init_configurations()
{
    get_mpi_config();
}

void set_mpi_info_config(MPI_Info info)
{

    char zerobuf[MPI_MAX_INFO_VAL];
    memset(zerobuf, 0, MPI_MAX_INFO_VAL);

    for (size_t key = 0; key < CONFIG_FIELD_COUNT; ++key)
    {
        cfg_opt_t opt = opts[key];
        if (memcmp(mpi_configs[key], zerobuf, MPI_MAX_INFO_VAL))
        {
            PMPI_Info_set(info, opt.name, mpi_configs[key]);
        }
    }
}

// TODO HDF5 Parameters