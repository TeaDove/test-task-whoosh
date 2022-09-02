## Test task Whoosh
## Описание задачи
Даны номера всех мобильных телефонов РФ<br>
Номера представляют собой int формата 99_999_999(так как первые 3 символа, 
+79, инвариантны для мобильных номеров)<br>
Необходимо их отсортировать максимально быстро по скорости<br>
Ограничения:
- Не учитывается тот факт, что они представляют собой натуральные числа 
от 0 до 999_999_999, сортировка подсчётом не используется.  
- Максимально используемая память - 256MB, постоянную память можно использовать хоть сколько
- 8 ядер процессора. 

## Описание решения
- Переводим файл с номерами телефонов из .txt формата в memmap
- Проводим первичную сортировку(radix sort), то есть разбиваем файл на части, который сортируем 
  отдельно и записываем в файлы. Сохраняем размер и имя файлов в PQueue(очередь, безопасная 
  для мультипроцессинга)
- Производим процедуру слияния созданных ранее файлов.
## Результаты
- Количество номеров: 1_000_000_000
- Максимальное количество RAM: 256МБ
- Количество ядер: 8 
- Первичная сортировка: 0:00:45.734593 
- Слияние: 1:22:43.737505
- Суммарно: 1:23:29.472099 (1 час, 23 минуты, 29 секунд)

## Установка зависимостей
``` bash
cd test-task-whoosh
python3 -m venv .venv
pip install poetry
poetry install
```
## Опции 
```bash
  -h, --help         show this help message and exit
  --generate         генерирует файл на 1ккк номеров телефонов
  --preprocess       переводит файл телефонов из .txt в np.memmap
  --generate_memmap  генерирует телефоны в memmap
  --sort             сортирует сгенерированный ранее файл
```
## Запуск
- В файле `app/__main__.py` выберем необходимые значения максимального количество ОЗУ 
  и номеров телефона. Для отладки и тестирования рекомендую использовать
  ```python
    MAX_RAM = 20
    NUMBERS_AMOUNT = 10_000_000
    ```
- Вначале необходимо сгенерировать номера телефонов<br>
`python3 -m app --generate`
- Далее переведём их в memmap<br>
`python3 -m app --preprocess`
- Далее создадим папку data и отсортируем <br>
`mkdir data; rm data/*; python3 -m app --sort`
## show_mem.py
Для валидации можно использовать скрипт `show_mem.py`, он выведет и минимальное, 
максимальное значение файла, его размер и количество ненулевых значений.<br>
Для простоты используйте:<br>
```bash
./app/show_mem.py data/*.dat
```
## Улучшения
Алгоритм не идеален, возможные улучшения:
- Сортировать без препроцессинга(то есть перевода из .txt в memmap)
