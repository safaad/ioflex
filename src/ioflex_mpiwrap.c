#include <dlfcn.h>
#include "ioflex_common.h"

FORWARD_DECL(PMPI_File_open, int, (MPI_Comm comm, const char *filename, int amode, MPI_Info info, MPI_File *fh));
FORWARD_DECL(PMPI_File_set_view, int, (MPI_File fh, MPI_Offset disp, MPI_Datatype etype, MPI_Datatype filetype, const char *datarep, MPI_Info info));
FORWARD_DECL(PMPI_File_set_info, int, (MPI_File fh, MPI_Info info));


static int rank = 0, mpi_rank_size = 0;
static MPI_Comm mpi_comm;

int DECL(MPI_File_open)(MPI_Comm comm, const char *filename, int amode, MPI_Info info, MPI_File *fh)
{

    int ret;

    MAP_OR_FAIL(PMPI_File_open);
    if (info == MPI_INFO_NULL)
        PMPI_Info_create(&info);
    // set info hints
    set_mpi_info_config(info);
    ret = __act_PMPI_File_open(comm, filename, amode, info, fh);
    return ret;
}

IOFLEX_WRAPPER_MAP(PMPI_File_open, int,  (MPI_Comm comm, const char *filename, int amode, MPI_Info info, MPI_File *fh), MPI_File_open)


int DECL(MPI_File_set_view)(MPI_File fh, MPI_Offset disp, MPI_Datatype etype, MPI_Datatype filetype, const char *datarep, MPI_Info info)
{
    int ret;

    MAP_OR_FAIL(PMPI_File_set_view);
    if (info == MPI_INFO_NULL)
        PMPI_Info_create(&info);
    set_mpi_info_config(info);
    ret = __act_PMPI_File_set_view(fh, disp, etype, filetype, datarep, info);
    return ret;
}

IOFLEX_WRAPPER_MAP(PMPI_File_set_view, int, (MPI_File fh, MPI_Offset disp, MPI_Datatype etype, MPI_Datatype filetype, const char *datarep, MPI_Info info), MPI_File_set_view)

int DECL(MPI_File_set_info)(MPI_File fh, MPI_Info info) {

    int ret;

    MAP_OR_FAIL(PMPI_File_set_info);
    if (info == MPI_INFO_NULL)
        PMPI_Info_create(&info);
    set_mpi_info_config(info);
    ret = __act_PMPI_File_set_info(fh, info);

    return ret;

}

IOFLEX_WRAPPER_MAP(PMPI_File_set_info, int, (MPI_File fh, MPI_Info info), MPI_File_set_info)