import argparse
import sys
import multiprocessing as mp
from pathlib import Path
from typing import Tuple
import os


from loguru import logger
import numpy as np
from tqdm import tqdm

# Максимальный размер потраченной ОЗУ одновременно в МБ
MAX_RAM = 1_024
MAX_RAM_BYTE = MAX_RAM * 1024 * 1024
NUMBERS_AMOUNT = 1_000_000_000


def generate():
    # TODO генерировать массив в зависимости от имеющейся озу
    open("numbers.txt", 'w')  # перезаписываем файл
    with open('numbers.txt', 'a') as f:
        for idx in np.arange(1_000):
            to_f = np.random.randint(0, 1_000_000_000, 1_000_000)
            f.write('\n'.join(map(str, to_f)))
            if idx % 50 == 0:
                logger.info("Сгенерировано: {} из {}", (idx+1)*1_000_000, NUMBERS_AMOUNT)


def preprocess():
    array = np.zeros((NUMBERS_AMOUNT,), dtype='int32')
    with open('numbers.txt', 'r') as f:
        for jdx in tqdm(range(NUMBERS_AMOUNT)):
            array[jdx] = (int(f.readline()))

    fp = np.memmap('tmp/numbers.dat', 'int32', 'w+', shape=(NUMBERS_AMOUNT,))
    fp[:] = array[:]
    fp.flush()


def generate_memmap():
    fp = np.memmap('tmp/numbers.dat', 'int32', 'w+', shape=(NUMBERS_AMOUNT,))

    for idx in tqdm(np.arange(100)):
        to_f = np.random.randint(0, 1_000_000_000, 10_000_000)
        fp[idx*10_000_000:(idx+1)*10_000_000] = to_f
        fp.flush()


def sort_worker(numbers: np.array, idx: int):
    numbers.sort()

    fp = np.memmap(f'data/{idx}.dat', dtype='int32', mode='w+', shape=numbers.shape)
    fp[:] = numbers[:]
    fp.flush()

    logger.info("Процесс {} закончен", idx)


def merger_worker(array_tuple1: Tuple[str, int], array_tuple2: Tuple[str, int], max_numbers: int, pqueue: mp.Queue):
    filename1, filesize1 = array_tuple1
    filename2, filesize2 = array_tuple2
    fp1 = np.memmap('data/' + filename1 + '.dat', dtype='int32', mode='r', shape=(max_numbers // 4, ))
    fp2 = np.memmap('data/' + filename2 + '.dat', dtype='int32', mode='r', shape=(max_numbers // 4, ))

    fp_out = np.memmap("data/{}_{}.dat".format(filename1, filename2), dtype='int32', mode='w+',
                       shape=(max_numbers // 2, ))

    index_fp1, index_fp2 = 0, 0
    while True:
        idx1, idx2, jdx = 0, 0, 0
        len_fp1, len_fp2 = len(fp1), len(fp2)
        while idx1 < len_fp1 and idx2 < len_fp2:
            if fp1[idx1] > fp2[idx2]:
                fp_out[jdx] = fp2[idx2]
                idx2 += 1
            else:
                fp_out[jdx] = fp1[idx1]
                idx1 += 1
            jdx += 1
        if idx1 < len_fp1:
            fp_out[jdx:] = fp1[idx1:]
        if idx2 < len_fp2:
            fp_out[jdx:] = fp2[idx2:]
        fp_out.flush()

        if index_fp1 * max_numbers >= filesize1 or index_fp2 * max_numbers >= filesize2:
            break
        fp1 = np.memmap('data/' + filename1 + '.dat', dtype='int32', mode='r', offset=(index_fp1*max_numbers),
                        shape=(max_numbers // 4,))
        fp2 = np.memmap('data/' + filename2 + '.dat', dtype='int32', mode='r', offset=(index_fp2 * max_numbers),
                        shape=(max_numbers // 4,))

        fp_out = np.memmap("data/{}_{}.dat".format(filename1, filename2), dtype='int32', mode='r+',
                           offset=((index_fp1 + index_fp2) * max_numbers * 2), shape=(max_numbers // 2,))
        index_fp1 += 1
        index_fp2 += 1

    while index_fp1 * max_numbers < filesize1:
        fp1 = np.memmap('data/' + filename1 + '.dat', dtype='int32', mode='r', offset=(index_fp1 * max_numbers),
                        shape=(max_numbers // 4,))
        fp_out = np.memmap("data/{}_{}.dat".format(filename1, filename2), dtype='int32', mode='r+',
                           offset=((index_fp1 + index_fp2) * max_numbers * 2), shape=(max_numbers // 2,))

        fp_out[:] = fp1[:]
        fp_out.flush()
        index_fp1 += 1

    while index_fp2 * max_numbers < filesize2:
        fp2 = np.memmap('data/' + filename2 + '.dat', dtype='int32', mode='r', offset=(index_fp2 * max_numbers),
                        shape=(max_numbers // 4,))
        fp_out = np.memmap("data/{}_{}.dat".format(filename1, filename2), dtype='int32', mode='r+',
                           offset=((index_fp1 + index_fp2) * max_numbers * 2), shape=(max_numbers // 2,))

        fp_out[:] = fp2[:]
        fp_out.flush()
        index_fp2 += 1

    os.remove('data/' + filename1 + '.dat')
    os.remove('data/' + filename2 + '.dat')
    logger.info("{}_{} готово", filename1, filename2)
    logger.info(fp_out)
    pqueue.put((filename1 + "_" + filename2, len_fp1+len_fp2))


def sort_():
    cpu_count = mp.cpu_count() - 1
    logger.info("Количество доступных ядер: {} ", cpu_count)
    logger.info("Количество доступной ОЗУ всего: {} ГБ", MAX_RAM)
    """
    1 телефон занимает 32 бита, 1 массив np.array - 104 + 4 * n (байт), (8*4 = 32) 
    """
    phones_per_batch = int(np.ceil((MAX_RAM_BYTE - 104) / 4 / cpu_count))
    bathes_amount = int(np.ceil(1_000_000_000 / phones_per_batch))
    logger.info("Количество телефонов в одном батче: {}", phones_per_batch)
    logger.info("Количество батчей всего: {}", bathes_amount)

    pool = mp.Pool(processes=cpu_count)
    mp_manager = mp.Manager()
    pqueue = mp_manager.Queue()

    for idx in range(bathes_amount-1):
        # fp = np.memmap('tmp/numbers.dat', mode='c', dtype='int32', offset=4*idx*phones_per_batch,
        #                shape=(phones_per_batch, ))
        #
        # pool.apply_async(sort_worker, args=(fp, idx))
        pqueue.put((str(idx), phones_per_batch))
        logger.info("Процесс {} стартовал", idx)

    # fp = np.memmap('tmp/numbers.dat', mode='c', dtype='int32', offset=4 * (idx+1) * phones_per_batch)
    # pool.apply_async(sort_worker, args=(fp, idx+1))
    pqueue.put((str(idx), NUMBERS_AMOUNT-(4 * (idx+1) * phones_per_batch)))
    # logger.info("Процесс {} стартовал", idx + 1)

    pool.close()
    pool.join()

    max_numbers_per_cpu = int(np.ceil((MAX_RAM_BYTE - 312) / cpu_count / 4))
    pool2 = mp.Pool(processes=cpu_count)
    while len(list(Path('data').iterdir())) > 1:
        f1, f2 = pqueue.get(block=True), pqueue.get(block=True)
        logger.info("{}_{} стартовал", f1[0], f2[0])
        pool2.apply_async(merger_worker, (f1, f2, max_numbers_per_cpu, pqueue))

    pool2.close()
    pool2.join()


def main():
    parser = argparse.ArgumentParser(description='тестовое задание для собеседования в whoosh')
    parser.add_argument('--generate', action="store_true", help='только генерирует файл на 1ккк номеров телефонов')
    parser.add_argument('--preprocess', action="store_true", help='только переводит файл телефонов из .txt в np.memmap')
    parser.add_argument('--generate_memmap', action="store_true", help='сразу генерирует телефоны в memmap')
    parser.add_argument('--sort', action="store_true", help='только сортирует сгенерированный ранее файл')
    args = parser.parse_args()

    if args.generate:
        generate()
        sys.exit()
    if args.sort:
        sort_()
        sys.exit()
    if args.preprocess:
        preprocess()
        sys.exit()
    if args.generate_memmap:
        generate_memmap()
        sys.exit()
    generate()
    sort_()
    sys.exit()




if __name__ == "__main__":
    main()