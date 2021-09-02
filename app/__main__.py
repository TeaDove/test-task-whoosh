import argparse
import sys
import multiprocessing as mp
from pathlib import Path
from typing import Tuple
import os


from loguru import logger
import numpy as np
from tqdm import tqdm

from .merger import merge_main

# Максимальный размер доступной ОЗУ одновременно в МБ
MAX_RAM = 1
MAX_RAM_BYTE = MAX_RAM * 1024 * 1024
NUMBERS_AMOUNT = 1_000_000


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
    numbers_per_iter = int(NUMBERS_AMOUNT // 100)
    for idx in tqdm(np.arange(100)):
        to_f = np.random.randint(0, 1_000_000_000, numbers_per_iter)
        fp[idx*numbers_per_iter:(idx+1)*numbers_per_iter] = to_f
        fp.flush()


def sort_worker(numbers: np.array, idx: int):
    numbers.sort()

    fp = np.memmap(f'data/{idx}.dat', dtype='int32', mode='w+', shape=numbers.shape)
    fp[:] = numbers[:]
    fp.flush()

    logger.info("Процесс {} закончен", idx)


def sort_():
    cpu_count = mp.cpu_count() - 1
    cpu_count = 2
    logger.info("Количество доступных ядер: {} ", cpu_count)
    logger.info("Количество доступной ОЗУ всего: {} МБ", MAX_RAM)
    """
    1 телефон занимает 32 бита, 1 массив np.array - 104 + 4 * n (байт), (8*4 = 32) 
    """
    phones_per_batch = int(np.ceil((MAX_RAM_BYTE - 104) / 4 / cpu_count))
    bathes_amount = int(np.ceil(NUMBERS_AMOUNT / phones_per_batch))
    logger.info("Количество телефонов в одном батче: {}", phones_per_batch)
    logger.info("Количество батчей всего: {}", bathes_amount)

    pool = mp.Pool(processes=cpu_count)
    mp_manager = mp.Manager()
    pqueue = mp_manager.Queue()

    # bathes_amount = 10
    idx = 0
    for idx in range(bathes_amount-1):
        fp = np.memmap('tmp/numbers.dat', mode='c', dtype='int32', offset=4*idx*phones_per_batch,
                       shape=(phones_per_batch, ))

        pool.apply_async(sort_worker, args=(fp, idx))
        pqueue.put((str(idx), phones_per_batch))
        logger.info("Процесс {} стартовал", idx)

    fp = np.memmap('tmp/numbers.dat', mode='c', dtype='int32', offset=4 * (idx+1) * phones_per_batch)
    pool.apply_async(sort_worker, args=(fp, idx+1))
    pqueue.put((str(idx+1), NUMBERS_AMOUNT-((idx+1) * phones_per_batch)))
    logger.info("Процесс {} стартовал", idx + 1)

    pool.close()
    pool.join()

    merge_main(MAX_RAM_BYTE, cpu_count, pqueue)


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