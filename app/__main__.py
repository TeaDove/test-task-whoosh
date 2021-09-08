import argparse
import sys
import multiprocessing as mp
import time
from datetime import timedelta

from loguru import logger
import numpy as np
from tqdm import tqdm

from .merger import merge_main

# Максимальный размер доступной ОЗУ одновременно в МБ
MAX_RAM = 20
MAX_RAM_BYTE = MAX_RAM * 1024 * 1024
# Количество номеров. 1_000_000 сортируется за несколько секунд.
NUMBERS_AMOUNT = 10_000_000


def generate():
    """
    Генерация NUMBERS_AMOUNT телефонов в файл в формат .txt
    """
    # TODO генерировать массив в зависимости от имеющейся озу
    start = time.time()
    open("numbers.txt", 'w')  # перезаписываем файл
    with open('numbers.txt', 'a') as f:
        for _ in tqdm(np.arange(NUMBERS_AMOUNT)):
            to_f = np.random.randint(0, 1_000_000_000)
            f.write(f"{to_f}\n")
        f.write('\n')
    all_time = timedelta(seconds=time.time() - start)
    logger.warning("Затраты по времени на генерацию: {}", all_time)


def preprocess():
    """
    Перевод выше сгенерированного файла в numpy.memmap(то есть отображения np.array в файле)
    """
    start = time.time()
    fp = np.memmap('tmp/numbers.dat', 'int32', 'write', shape=(NUMBERS_AMOUNT,))
    with open('numbers.txt', 'r') as f:
        for jdx in tqdm(range(NUMBERS_AMOUNT)):
            fp[jdx] = int(f.readline())
    fp.flush()
    all_time = timedelta(seconds=time.time() - start)
    logger.warning("Затраты по времени на препроцессинг: {}", all_time)


def generate_memmap():
    """
    Генерация меммапа, нужно для отладки, чтобы не генерировать каждый раз файл и не переводить в memmap
    """
    fp = np.memmap('tmp/numbers.dat', 'int32', 'w+', shape=(NUMBERS_AMOUNT,))
    numbers_per_iter = int(NUMBERS_AMOUNT // 100)
    for idx in tqdm(np.arange(100)):
        to_f = np.random.randint(0, 1_000_000_000, numbers_per_iter)
        fp[idx*numbers_per_iter:(idx+1)*numbers_per_iter] = to_f
        fp.flush()


def sort_worker(numbers: np.array, idx: int):
    """
    Функция процесса первичной сортировки
    :param numbers: сами числа для сортировки в формате np.array
    :param idx: индекс сортируемой части чисел
    """
    numbers.sort(kind='stable') # stable - под катом используется radix сорт, что быстра в данной задаче

    fp = np.memmap(f'data/{idx}.dat', dtype='int32', mode='w+', shape=numbers.shape)
    fp[:] = numbers[:]
    fp.flush()

    logger.info("Процесс {} закончен", idx)


def sort_():
    """
    Главный метод сортировки
    """
    cpu_count = mp.cpu_count() - 1
    logger.info("Количество доступных ядер: {} ", cpu_count)
    logger.info("Количество доступной ОЗУ всего: {} МБ", MAX_RAM)
    """
    1 телефон занимает 32 бита, 1 массив np.array - 104 + 4 * n (байт). 104 на сам массив, 4 байта на 1 int32.  
    """
    phones_per_batch = int(np.ceil((MAX_RAM_BYTE - 104) / 4 / cpu_count))
    bathes_amount = int(np.ceil(NUMBERS_AMOUNT / phones_per_batch))
    logger.info("Количество телефонов в одном батче: {}", phones_per_batch)
    logger.info("Количество батчей всего: {}", bathes_amount)

    prime_sort_start = time.time()
    pool = mp.Pool(processes=cpu_count)
    mp_manager = mp.Manager()
    pqueue = mp_manager.Queue()

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
    prime_sort_stop = time.time()

    merge_start = time.time()
    merge_main(MAX_RAM_BYTE, cpu_count, pqueue)
    merge_stop = time.time()
    merge_time = timedelta(seconds=merge_stop - merge_start)
    sort_time = timedelta(seconds=prime_sort_stop - prime_sort_start)
    all_time = timedelta(seconds=merge_stop - prime_sort_start)
    logger.warning("Затраты по времени на сортировку:")
    logger.warning("Первичная сортировка: {}, слияние: {}, суммарно: {}",sort_time, merge_time, all_time)

def main():
    """
    Обработка опций
    """
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
    preprocess()
    sort_()
    sys.exit()




if __name__ == "__main__":
    main()