import darshan
import pandas as pd




def merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res


def parsedarshan(logfile):
    
    posix_time_list = ["POSIX_F_READ_TIME", "POSIX_F_WRITE_TIME", "POSIX_F_META_TIME"]
    posix_list = [
        "POSIX_OPENS",
        "POSIX_WRITES",
        "POSIX_READS",
        "POSIX_STATS",
        "POSIX_SEEKS",
        "POSIX_MMAPS",
        "POSIX_FSYNCS",
        "POSIX_CONSEC_READS",
        "POSIX_CONSEC_WRITES",
        "POSIX_SEQ_READS",
        "POSIX_SEQ_WRITES",
        "POSIX_SIZE_READ_0_100",
        "POSIX_SIZE_WRITE_0_100",
        "POSIX_SIZE_READ_100_1K",
        "POSIX_SIZE_WRITE_100_1K",
        "POSIX_SIZE_READ_1K_10K",
        "POSIX_SIZE_WRITE_1K_10K",
        "POSIX_SIZE_READ_10K_100K",
        "POSIX_SIZE_WRITE_10K_100K",
        "POSIX_SIZE_READ_100K_1M",
        "POSIX_SIZE_WRITE_100K_1M",
        "POSIX_SIZE_READ_1M_4M",
        "POSIX_SIZE_WRITE_1M_4M",
        "POSIX_SIZE_READ_4M_10M",
        "POSIX_SIZE_WRITE_4M_10M",
        "POSIX_SIZE_READ_10M_100M",
        "POSIX_SIZE_WRITE_10M_100M",
        "POSIX_SIZE_READ_100M_1G",
        "POSIX_SIZE_WRITE_100M_1G",
        "POSIX_SIZE_READ_1G_PLUS",
        "POSIX_SIZE_WRITE_1G_PLUS",
        "POSIX_BYTES_READ",
        "POSIX_BYTES_WRITTEN",
    ]

    mpiio_time_list = ["MPIIO_F_READ_TIME", "MPIIO_F_WRITE_TIME", "MPIIO_F_META_TIME"]
    mpiio_list = [
        "MPIIO_INDEP_OPENS",
        "MPIIO_COLL_OPENS",
        "MPIIO_INDEP_WRITES",
        "MPIIO_COLL_WRITES",
        "MPIIO_INDEP_READS",
        "MPIIO_COLL_READS",
        "MPIIO_SYNCS",
        "MPIIO_SIZE_READ_AGG_0_100",
        "MPIIO_SIZE_WRITE_AGG_0_100",
        "MPIIO_SIZE_READ_AGG_100_1K",
        "MPIIO_SIZE_WRITE_AGG_100_1K",
        "MPIIO_SIZE_READ_AGG_1K_10K",
        "MPIIO_SIZE_WRITE_AGG_1K_10K",
        "MPIIO_SIZE_READ_AGG_10K_100K",
        "MPIIO_SIZE_WRITE_AGG_10K_100K",
        "MPIIO_SIZE_READ_AGG_100K_1M",
        "MPIIO_SIZE_WRITE_AGG_100K_1M",
        "MPIIO_SIZE_READ_AGG_1M_4M",
        "MPIIO_SIZE_WRITE_AGG_1M_4M",
        "MPIIO_SIZE_READ_AGG_4M_10M",
        "MPIIO_SIZE_WRITE_AGG_4M_10M",
        "MPIIO_SIZE_READ_AGG_10M_100M",
        "MPIIO_SIZE_WRITE_AGG_10M_100M",
        "MPIIO_SIZE_READ_AGG_100M_1G",
        "MPIIO_SIZE_WRITE_AGG_100M_1G",
        "MPIIO_SIZE_READ_AGG_1G_PLUS",
        "MPIIO_SIZE_WRITE_AGG_1G_PLUS",
        "MPIIO_BYTES_READ",
        "MPIIO_BYTES_WRITTEN",
    ]

    report = darshan.DarshanReport(logfile, read_all=True)
    report.info()
    runtime = report.metadata['job']['run_time']
    nprocs = report.metadata["job"]["nprocs"]

    print(runtime)
    dict1 = {"nprocs": nprocs, "runtime": runtime}
    if "POSIX" in report.modules:
        df = report.records["POSIX"].to_df()
        posix_df = pd.merge(
            df["counters"],
            df["fcounters"],
            left_on=["id", "rank"],
            right_on=["id", "rank"],
        )
        dict1 = merge(posix_df[posix_list].sum(axis=0).divide(nprocs).to_dict(), dict1)
        dict1 = merge(
            posix_df[posix_time_list].sum(axis=0).divide(nprocs * runtime).to_dict(),
            dict1,
        )

    else:
        posix_list += posix_time_list
        for key in posix_list:
            dict1[key] = 0.0


    if "MPI-IO" in report.modules:
        df = report.records["MPI-IO"].to_df()
        mpiio_df = pd.merge(
            df["counters"],
            df["fcounters"],
            left_on=["id", "rank"],
            right_on=["id", "rank"],
        )
        dict1 = merge(mpiio_df[mpiio_list].sum(axis=0).divide(nprocs), dict1)
        dict1 = merge(
            mpiio_df[mpiio_time_list].sum(axis=0).divide(nprocs * runtime), dict1
        )


    else:
        mpiio_list += mpiio_time_list
        for key in mpiio_list:
            dict1[key] = 0.0


    if "POSIX" in report.modules:
        condition = "POSIX_OPENS > 0"
        t = posix_df.query(condition).groupby("id").max()
        c = t.shape[0]
        dict1["total_files"] = c

        condition = "(POSIX_READS > 0 & POSIX_WRITES <= 0)"
        t = posix_df.query(condition).groupby("id").max()
        c = t.shape[0]
        dict1["nb_read-only_files"] = c

        condition = "(POSIX_WRITES > 0 & POSIX_READS <= 0)"
        t = posix_df.query(condition).groupby("id").max()
        c = t.shape[0]
        dict1["nb_write-only_files"] = c

        condition = "(POSIX_WRITES > 0 & POSIX_READS > 0)"
        t = posix_df.query(condition).groupby("id").max()
        c = t.shape[0]
        dict1["nb_read-write_files"] = c
    else:
        dict1["total_files"] = 0
        dict1["nb_read-only_files"] = 0
        dict1["nb_write-only_files"] = 0
        dict1["nb_read-write_files"] = 0

    return dict1
