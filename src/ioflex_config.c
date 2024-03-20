#include <confuse.h>
#include "ioflex_common.h"

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

void get_mpi_config(mpi_configs *mpi_configs)
{

    cfg_t *cfg;
    parse_config_file(&cfg);

    if (cfg_getstr(cfg, "romio_ds_read"))
        strcpy(mpi_configs->romio_ds_read, cfg_getstr(cfg, "romio_ds_read"));
    if (cfg_getstr(cfg, "romio_ds_write"))
        strcpy(mpi_configs->romio_ds_write, cfg_getstr(cfg, "romio_ds_write"));
    if (cfg_getstr(cfg, "romio_cb_read"))
        strcpy(mpi_configs->romio_cb_read, cfg_getstr(cfg, "romio_cb_read"));
    if (cfg_getstr(cfg, "romio_cb_write"))
        strcpy(mpi_configs->romio_cb_write, cfg_getstr(cfg, "romio_cb_write"));
    if (cfg_getstr(cfg, "romio_filesystem_type"))
        strcpy(mpi_configs->romio_filesystem_type, cfg_getstr(cfg, "romio_filesystem_type"));
    if (cfg_getstr(cfg, "cb_config_list"))
        strcpy(mpi_configs->cb_config_list, cfg_getstr(cfg, "cb_config_list"));
    if (cfg_getstr(cfg, "striping_factor"))
        strcpy(mpi_configs->striping_factor, cfg_getstr(cfg, "striping_factor"));
    if (cfg_getstr(cfg, "striping_unit"))
        strcpy(mpi_configs->striping_unit, cfg_getstr(cfg, "striping_unit"));
    if (cfg_getstr(cfg, "cb_nodes"))
        strcpy(mpi_configs->cb_nodes, cfg_getstr(cfg, "cb_nodes"));
    if (cfg_getstr(cfg, "cb_buffer_size"))
        strcpy(mpi_configs->cb_buffer_size, cfg_getstr(cfg, "cb_buffer_size"));

    cfg_free(cfg);
}

void set_mpi_info_config(MPI_Info info)
{

    mpi_configs mpi_configs;
    memset(&mpi_configs, 0, sizeof(mpi_configs));
    get_mpi_config(&mpi_configs);

    char zerobuf[MPI_MAX_INFO_VAL];
    memset(zerobuf, 0, MPI_MAX_INFO_VAL);

    if (memcmp(mpi_configs.romio_ds_read, zerobuf, MPI_MAX_INFO_VAL))
        MPI_Info_set(info, "romio_ds_read", mpi_configs.romio_ds_read);
    if (memcmp(mpi_configs.romio_ds_write, zerobuf, MPI_MAX_INFO_VAL))
        MPI_Info_set(info, "romio_ds_write", mpi_configs.romio_ds_write);
    if (memcmp(mpi_configs.romio_cb_read, zerobuf, MPI_MAX_INFO_VAL))
        MPI_Info_set(info, "romio_cb_read", mpi_configs.romio_cb_read);
    if (memcmp(mpi_configs.romio_cb_write, zerobuf, MPI_MAX_INFO_VAL))
        MPI_Info_set(info, "romio_cb_write", mpi_configs.romio_cb_write);
    if (memcmp(mpi_configs.romio_filesystem_type, zerobuf, MPI_MAX_INFO_VAL))
        MPI_Info_set(info, "romio_filesystem_type", mpi_configs.romio_filesystem_type);
    if (memcmp(mpi_configs.cb_config_list, zerobuf, MPI_MAX_INFO_VAL))
        MPI_Info_set(info, "cb_config_list", mpi_configs.cb_config_list);
    if (memcmp(mpi_configs.striping_unit, zerobuf, MPI_MAX_INFO_VAL))
        MPI_Info_set(info, "striping_unit", mpi_configs.striping_unit);
    if (memcmp(mpi_configs.striping_factor, zerobuf, MPI_MAX_INFO_VAL))
        MPI_Info_set(info, "striping_factor", mpi_configs.striping_factor);
    if (memcmp(mpi_configs.cb_nodes, zerobuf, MPI_MAX_INFO_VAL))
        MPI_Info_set(info, "cb_nodes", mpi_configs.cb_nodes);
    if (memcmp(mpi_configs.cb_buffer_size, zerobuf, MPI_MAX_INFO_VAL))
        MPI_Info_set(info, "cb_buffer_size", mpi_configs.cb_buffer_size);
}

// TODO HDF5 Parameters