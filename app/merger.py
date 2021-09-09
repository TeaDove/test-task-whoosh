import os
import multiprocessing as mp
from pathlib import Path
from typing import Tuple
import time

from loguru import logger
import numpy as np


def merger_worker(array_tuple1: Tuple[str, int], array_tuple2: Tuple[str, int], max_numbers: int, pqueue: mp.Queue):
    """
    Процесс слияния меммапов
    :param array_tuple1: первый массив
    :param array_tuple2: второй массив
    :param max_numbers: максимальное количество номеров, что можно одновременно открыть
    :param pqueue: очередь безопасная для мультипроцессинга
    """
    filename1, filesize1 = array_tuple1
    filename2, filesize2 = array_tuple2
    filesize_out = filesize1 + filesize2
    idx1, idx2, idx_out = 0, 0, 0

    fp1 = np.memmap('data/' + filename1 + '.dat', dtype='int32', mode='r', shape=(max_numbers // 4, ))
    fp2 = np.memmap('data/' + filename2 + '.dat', dtype='int32', mode='r', shape=(max_numbers // 4, ))

    # FIXME заменить генерацию имени на последовательную нумерацию, которая берётся из PQueue. 
    filename_out = str(int(time.time() * 10e6))
    fp_out = np.memmap("data/{}.dat".format(filename_out), dtype='int32', mode='w+',
                       shape=(max_numbers // 2, ))
    file_offset_1, file_offset_2, file_offset_out = fp1.shape[0], fp2.shape[0], fp_out.shape[0]
    len_fp1, len_fp2, len_fp_out = len(fp1), len(fp2), len(fp_out)

    def open_new_out():
        """
        Открывает новый memmap для out файла
        """
        nonlocal fp_out, file_offset_out, idx_out, len_fp_out, filesize_out, filename_out
        fp_out.flush()
        fp_shape = min(max_numbers // 2, (filesize_out - file_offset_out))

        fp_out = np.memmap("data/{}.dat".format(filename_out), dtype='int32', mode='r+',
                           offset=(file_offset_out * 4),
                           shape=(fp_shape,))
        file_offset_out += fp_shape
        idx_out = 0
        len_fp_out = len(fp_out)

    while file_offset_1 < filesize1 and file_offset_2 < filesize2:
        """
        Основной цикл сортировки, в нём проверяет какой элемент массивов меньше, он добавляется в новый массив
        Если какой-то из массивов кончается, файл закрываеться, открывается новый и массив заполняется новыми 
        элементами
        """
        if fp1[idx1] > fp2[idx2]:
            fp_out[idx_out] = fp2[idx2]
            idx2 += 1
        else:
            fp_out[idx_out] = fp1[idx1]
            idx1 += 1
        idx_out += 1

        if idx_out >= len_fp_out:
            open_new_out()

        if idx1 >= len_fp1:
            file_shape = min(max_numbers // 4, (filesize1 - file_offset_1))
            fp1 = np.memmap('data/' + filename1 + '.dat', dtype='int32', mode='r', offset=(file_offset_1 * 4),
                            shape=(file_shape, ))
            len_fp1 = len(fp1)
            idx1 = 0
            file_offset_1 += file_shape

        if idx2 >= len_fp2:
            file_shape = min(max_numbers // 4, (filesize2 - file_offset_2))
            fp2 = np.memmap('data/' + filename2 + '.dat', dtype='int32', mode='r', offset=(file_offset_2 * 4),
                            shape=(file_shape, ))
            len_fp2 = len(fp2)
            idx2 = 0
            file_offset_2 += file_shape

    while idx1 < len_fp1 and idx2 < len_fp2:
        """
        Дозаполнение массива
        """
        if fp1[idx1] > fp2[idx2]:
            fp_out[idx_out] = fp2[idx2]
            idx2 += 1
        else:
            fp_out[idx_out] = fp1[idx1]
            idx1 += 1
        idx_out += 1

        if idx_out >= len_fp_out:
            open_new_out()

    while file_offset_1 < filesize1 or idx1 < len_fp1:
        """
        Цикл дозаполнения из массива 1. (ниже для массива 2)
        """
        if idx_out >= len_fp_out:
            open_new_out()
        if idx1 >= len_fp1:
            file_shape = min(max_numbers // 4, (filesize1 - file_offset_1))
            fp1 = np.memmap('data/' + filename1 + '.dat', dtype='int32', mode='r', offset=(file_offset_1 * 4),
                            shape=(file_shape,))
            len_fp1 = len(fp1)
            idx1 = 0
            file_offset_1 += file_shape
        len_to_flush = min(len_fp_out - idx_out, len_fp1 - idx1)
        fp_out[idx_out:len_to_flush + idx_out] = fp1[idx1: len_to_flush + idx1]
        idx1 += len_to_flush
        idx_out += len_to_flush

    while file_offset_2 < filesize2 or idx2 < len_fp2:
        if idx_out >= len_fp_out:
            open_new_out()
        if idx2 >= len_fp2:
            file_shape = min(max_numbers // 4, (filesize2 - file_offset_2))
            fp2 = np.memmap('data/' + filename2 + '.dat', dtype='int32', mode='r', offset=(file_offset_2 * 4),
                            shape=(file_shape,))
            len_fp2 = len(fp2)
            idx2 = 0
            file_offset_2 += file_shape
        len_to_flush = min(len_fp_out - idx_out, len_fp2 - idx2)
        fp_out[idx_out:len_to_flush + idx_out] = fp2[idx2: len_to_flush + idx2]
        idx2 += len_to_flush
        idx_out += len_to_flush

    # logger.info("Array idxs: {}/{}, {}/{}, {}/{}", idx1, len_fp1, idx2, len_fp2, idx_out, len_fp_out)
    # logger.info("File idxs: {}/{}, {}/{}, {}/{}", file_offset_1, filesize1, file_offset_2, filesize2, file_offset_out,
    #             filesize_out)

    fp_out.flush()
    os.remove('data/' + filename1 + '.dat')
    os.remove('data/' + filename2 + '.dat')
    # logger.info("{}\t{:,}\t{:,}\t{:,}", fp_out, fp_out.min(), fp_out.max(), fp_out.shape[0])
    logger.info("{} готов", filename_out)

    if len(list(Path('data').iterdir())) <= 1:
        # Останавливаем все процессы, если больше сортировать нечего
        pqueue.put(("Stop", ))
        pqueue.put(("Stop", ))
    else:
        pqueue.put((filename_out, filesize_out))


def error_handler(exception: BaseException):
    """
    Логер ошибок из пула мержда
    :param exception: ошибка
    """
    logger.critical(exception)


def merge_main(max_ram_byte: int, cpu_count: int, pqueue: mp.Queue):
    """
    Управляющая функция
    :param max_ram_byte: максимально кол-во памяти, что можно использовать одновременно в Байтах
    :param cpu_count: количество одновременных процессов
    :param pqueue: безопасная для мультипроцессинга очередь
    """
    max_numbers_per_cpu = int(np.ceil((max_ram_byte - 312) / cpu_count / 4))
    max_numbers_per_cpu -= max_numbers_per_cpu % 4

    pool = mp.Pool(processes=cpu_count)
    while True:
        f1, f2 = pqueue.get(block=True), pqueue.get(block=True)
        if f1[0] == "Stop":
            break
        logger.info("{} {} стартовали, файлов для сортировки: {}", f1[0], f2[0], len(list(Path('data').iterdir())))
        pool.apply_async(merger_worker, (f1, f2, max_numbers_per_cpu, pqueue), error_callback=error_handler)

    pool.close()
    pool.join()

    # logger.info("Сортировка завершена")
    # fp_out_name = str(list(Path('data').iterdir())[0])
    # fp = np.memmap(fp_out_name, 'int32', 'r')
    # logger.info("{}:\t{}\t{:,}\t{:,}\t{:,}\t{:,}", fp_out_name, fp, fp.min(), fp.max(), fp.shape[0], np.count_nonzero(fp))