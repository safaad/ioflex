#include <ioflex_common.h>
#include <dlfcn.h>

FORWARD_DECL(MPI_File_open, int, (MPI_Comm comm, const char *filename, int amode, MPI_Info info, MPI_File *fh));
FORWARD_DECL(MPI_File_set_view, int, (MPI_File fh, MPI_Offset disp, MPI_Datatype etype, MPI_Datatype filetype, const char *datarep, MPI_Info info));
FORWARD_DECL(MPI_File_set_info, int, (MPI_File fh, MPI_Info info));

static int rank = 0, mpi_rank_size = 0;
static MPI_Comm mpi_comm;

int DECL(MPI_File_open)(MPI_Comm comm, const char *filename, int amode, MPI_Info info, MPI_File *fh)
{

    int ret;

    mpi_comm = comm;
    MPI_Comm_rank(mpi_comm, &rank);
    MPI_Comm_size(mpi_comm, &mpi_rank_size);

    DLSYM(MPI_File_open);
    if (info == MPI_INFO_NULL)
        MPI_Info_create(&info);
    // set info hints
    set_mpi_info_config(info);
    ret = __real_MPI_File_open(comm, filename, amode, info, fh);
    return ret;
}

int DECL(MPI_File_set_view)(MPI_File fh, MPI_Offset disp, MPI_Datatype etype, MPI_Datatype filetype, const char *datarep, MPI_Info info)
{

    int ret;

    DLSYM(MPI_File_set_view);
    if (info == MPI_INFO_NULL)
        MPI_Info_create(&info);
    set_mpi_info_config(info);
    ret = __real_MPI_File_set_view(fh, disp, etype, filetype, datarep, info);
    return ret;
}

int DECL(MPI_File_set_info)(MPI_File fh, MPI_Info info) {

    int ret;

    DLSYM(MPI_File_set_info);
    if (info == MPI_INFO_NULL)
        MPI_Info_create(&info);
    set_mpi_info_config(info);
    ret = __real_MPI_File_set_info(fh, info);

    return ret;

}