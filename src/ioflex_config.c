#include <confuse.h>
#include "ioflex_common.h"

bool MPI_CONFIG_IS_INITIALIZED = false;
mpi_configs mpiconfigs;

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
    CFG_END()};

int parse_config_file(cfg_t **cfg)
{
    *cfg = cfg_init(opts, CFGF_NONE);
    if (cfg_parse(*cfg, "config.conf") == CFG_PARSE_ERROR)
    {
        printf("Failed parsing config file");
        exit(1);
    }
}

void get_mpi_config()
{

    cfg_t *cfg;
    parse_config_file(&cfg);

    if (cfg_getstr(cfg, "romio_ds_read"))
        strcpy(mpiconfigs.romio_ds_read, cfg_getstr(cfg, "romio_ds_read"));
    if (cfg_getstr(cfg, "romio_ds_write"))
        strcpy(mpiconfigs.romio_ds_write, cfg_getstr(cfg, "romio_ds_write"));
    if (cfg_getstr(cfg, "romio_cb_read"))
        strcpy(mpiconfigs.romio_cb_read, cfg_getstr(cfg, "romio_cb_read"));
    if (cfg_getstr(cfg, "romio_cb_write"))
        strcpy(mpiconfigs.romio_cb_write, cfg_getstr(cfg, "romio_cb_write"));
    if (cfg_getstr(cfg, "romio_filesystem_type"))
        strcpy(mpiconfigs.romio_filesystem_type, cfg_getstr(cfg, "romio_filesystem_type"));
    if (cfg_getstr(cfg, "cb_config_list"))
        strcpy(mpiconfigs.cb_config_list, cfg_getstr(cfg, "cb_config_list"));
    if (cfg_getstr(cfg, "striping_factor"))
        strcpy(mpiconfigs.striping_factor, cfg_getstr(cfg, "striping_factor"));
    if (cfg_getstr(cfg, "striping_unit"))
        strcpy(mpiconfigs.striping_unit, cfg_getstr(cfg, "striping_unit"));
    if (cfg_getstr(cfg, "cb_nodes"))
        strcpy(mpiconfigs.cb_nodes, cfg_getstr(cfg, "cb_nodes"));
    if (cfg_getstr(cfg, "cb_buffer_size"))
        strcpy(mpiconfigs.cb_buffer_size, cfg_getstr(cfg, "cb_buffer_size"));

    cfg_free(cfg);
}

void set_mpi_info_config(MPI_Info info)
{

    if (!MPI_CONFIG_IS_INITIALIZED)
    {
        memset(&mpiconfigs, 0, sizeof(mpiconfigs));
        get_mpi_config();
        MPI_CONFIG_IS_INITIALIZED = true;
    }
    char zerobuf[MPI_MAX_INFO_VAL];
    memset(zerobuf, 0, MPI_MAX_INFO_VAL);

    if (memcmp(mpiconfigs.romio_ds_read, zerobuf, MPI_MAX_INFO_VAL))
        PMPI_Info_set(info, "romio_ds_read", mpiconfigs.romio_ds_read);
    if (memcmp(mpiconfigs.romio_ds_write, zerobuf, MPI_MAX_INFO_VAL))
        PMPI_Info_set(info, "romio_ds_write", mpiconfigs.romio_ds_write);
    if (memcmp(mpiconfigs.romio_cb_read, zerobuf, MPI_MAX_INFO_VAL))
        PMPI_Info_set(info, "romio_cb_read", mpiconfigs.romio_cb_read);
    if (memcmp(mpiconfigs.romio_cb_write, zerobuf, MPI_MAX_INFO_VAL))
        PMPI_Info_set(info, "romio_cb_write", mpiconfigs.romio_cb_write);
    if (memcmp(mpiconfigs.romio_filesystem_type, zerobuf, MPI_MAX_INFO_VAL))
        PMPI_Info_set(info, "romio_filesystem_type", mpiconfigs.romio_filesystem_type);
    if (memcmp(mpiconfigs.cb_config_list, zerobuf, MPI_MAX_INFO_VAL))
        PMPI_Info_set(info, "cb_config_list", mpiconfigs.cb_config_list);
    if (memcmp(mpiconfigs.striping_unit, zerobuf, MPI_MAX_INFO_VAL))
        PMPI_Info_set(info, "striping_unit", mpiconfigs.striping_unit);
    if (memcmp(mpiconfigs.striping_factor, zerobuf, MPI_MAX_INFO_VAL))
        PMPI_Info_set(info, "striping_factor", mpiconfigs.striping_factor);
    if (memcmp(mpiconfigs.cb_nodes, zerobuf, MPI_MAX_INFO_VAL))
        PMPI_Info_set(info, "cb_nodes", mpiconfigs.cb_nodes);
    if (memcmp(mpiconfigs.cb_buffer_size, zerobuf, MPI_MAX_INFO_VAL))
        PMPI_Info_set(info, "cb_buffer_size", mpiconfigs.cb_buffer_size);
}

// TODO HDF5 Parameters