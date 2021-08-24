import os
import multiprocessing as mp
from pathlib import Path
from typing import Tuple

from loguru import logger
import numpy as np
from tqdm import tqdm


def merger_worker(array_tuple1: Tuple[str, int], array_tuple2: Tuple[str, int], max_numbers: int, pqueue: mp.Queue):
    filename1, filesize1 = array_tuple1
    filename2, filesize2 = array_tuple2
    fp_out_size = filesize1 + filesize2
    index_fp1, index_fp2, index_fp_out = 0, 0, 0
    idx1, idx2, jdx = 0, 0, 0

    # logger.info("{} {}", filesize1, filesize2)
    fp1 = np.memmap('data/' + filename1 + '.dat', dtype='int32', mode='r', shape=(max_numbers // 4, ))
    fp2 = np.memmap('data/' + filename2 + '.dat', dtype='int32', mode='r', shape=(max_numbers // 4, ))

    fp_out = np.memmap("data/{}_{}.dat".format(filename1, filename2), dtype='int32', mode='w+',
                       shape=(max_numbers // 2, ))

    len_fp1, len_fp2, len_fp_out = len(fp1), len(fp2), len(fp_out)
    while index_fp1 * max_numbers // 4 < filesize1 and index_fp2 * max_numbers // 4 < filesize2:
        # logger.info("{} {} {}", index_fp1, index_fp2, index_fp_out)
        # logger.info("{} {} {}", (1+index_fp1) * max_numbers // 4, (1+index_fp2) * max_numbers // 4, (1+index_fp_out) * max_numbers // 2)

        while idx1 < len_fp1 and idx2 < len_fp2 and jdx < len_fp_out:

            if fp1[idx1] > fp2[idx2]:
                fp_out[jdx] = fp2[idx2]
                idx2 += 1
            else:
                fp_out[jdx] = fp1[idx1]
                idx1 += 1
            jdx += 1

        # logger.info("{}\t{:,}\t{:,}\t{:,}", fp_out, fp_out.min(), fp_out.max(), fp_out.shape[0])

        if idx1 >= len_fp1:
            index_fp1 += 1
            if index_fp1 * max_numbers // 4 >= filesize1:
                break
            else:
                fp1 = np.memmap('data/' + filename1 + '.dat', dtype='int32', mode='r', offset=(index_fp1 * max_numbers // 4 * 4),
                                shape=(min(max_numbers // 4, (filesize1 - index_fp1 * max_numbers // 4)), ))
                len_fp1 = len(fp1)
                idx1 = 0

        if idx2 >= len_fp2:
            index_fp2 += 1
            if index_fp2 * max_numbers // 4 >= filesize2:
                break
            else:

                fp2 = np.memmap('data/' + filename2 + '.dat', dtype='int32', mode='r', offset=(index_fp2 * max_numbers // 4 * 4),
                                shape=(min(max_numbers // 4, (filesize2 - index_fp2 * max_numbers // 4)), ))
                len_fp2 = len(fp2)
                idx2 = 0

        if jdx >= len_fp_out:
            fp_out.flush()
            index_fp_out += 1
            fp_out = np.memmap("data/{}_{}.dat".format(filename1, filename2), dtype='int32', mode='r+',
                               offset=(index_fp_out * max_numbers * 2 // 4 * 4),
                               shape=(min(max_numbers // 2, (fp_out_size - index_fp_out * max_numbers // 2)), ))
            jdx = 0
            len_fp_out = len(fp_out)

    logger.info("{}/{}, {}/{}, {}/{}", idx1, len_fp1, idx2, len_fp2, jdx, len_fp_out)
    # if idx1 < len_fp1:
    #     to_out = min(len_fp_out - jdx, len_fp1 - idx1)
    #     fp_out[jdx:to_out-jdx] = fp1[idx1:to_out-idx1]
        # if idx1 + to_out < len_fp1:
        #     fp_out.flush()
        #     index_fp_out += 1
        #     fp_out = np.memmap("data/{}_{}.dat".format(filename1, filename2), dtype='int32', mode='r+',
        #                        offset=(index_fp_out * max_numbers * 2 // 4 * 4),
        #                        shape=(min(max_numbers // 2, (fp_out_size - index_fp_out * max_numbers // 2)),))
        #

    # while index_fp1 * max_numbers < filesize1:
    #
    #
    #
    #     fp1 = np.memmap('data/' + filename1 + '.dat', dtype='int32', mode='r', offset=(index_fp1 * max_numbers),
    #                     shape=(max_numbers // 4,))
    #     fp_out = np.memmap("data/{}_{}.dat".format(filename1, filename2), dtype='int32', mode='r+',
    #                        offset=((index_fp1 + index_fp2) * max_numbers * 2), shape=(max_numbers // 2,))
    #
    #     fp_out[:] = fp1[:]
    #     fp_out.flush()
    #     index_fp1 += 1
    #
    # while index_fp2 * max_numbers < filesize2:
    #     fp2 = np.memmap('data/' + filename2 + '.dat', dtype='int32', mode='r', offset=(index_fp2 * max_numbers),
    #                     shape=(max_numbers // 4,))
    #     fp_out = np.memmap("data/{}_{}.dat".format(filename1, filename2), dtype='int32', mode='r+',
    #                        offset=((index_fp1 + index_fp2) * max_numbers * 2), shape=(max_numbers // 2,))
    #
    #     fp_out[:] = fp2[:]
    #     fp_out.flush()
    #     index_fp2 += 1

    os.remove('data/' + filename1 + '.dat')
    os.remove('data/' + filename2 + '.dat')
    logger.info("{}\t{:,}\t{:,}\t{:,}", fp_out, fp_out.min(), fp_out.max(), fp_out.shape[0])
    logger.info("{}_{} готово", filename1, filename2)
    if len(list(Path('data').iterdir())) <= 1:
        pqueue.put(("Stop", ))
        pqueue.put(("Stop", ))
    else:
        pqueue.put((filename1 + "_" + filename2, filesize1 + filesize2))


def error_handler(exception: BaseException):
    logger.critical(exception)


def merge_main(max_ram_byte: int, cpu_count: int, pqueue: mp.Queue):
    max_numbers_per_cpu = int(np.ceil((max_ram_byte - 312) / cpu_count / 4))
    pool = mp.Pool(processes=cpu_count)
    while True:
        f1, f2 = pqueue.get(block=True), pqueue.get(block=True)
        if f1[0] == "Stop":
            break
        logger.info("{}_{} стартовал, {}", f1[0], f2[0], len(list(Path('data').iterdir())))
        pool.apply_async(merger_worker, (f1, f2, max_numbers_per_cpu, pqueue), error_callback=error_handler)

    pool.close()
    pool.join()

    logger.info("Сортировка завершена")
    fp_out_name = str(list(Path('data').iterdir())[0])
    fp = np.memmap(fp_out_name, 'int32', 'r')
    logger.info("{}:\t{}\t{:,}\t{:,}\t{:,}", fp_out_name, fp, fp.min(), fp.max(), fp.shape[0])