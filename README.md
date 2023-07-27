#  Проект "Физически-информированная модель нефтяного месторождения". Летняя школа AIRI-2023

Проект посвящен исследованию т.н. физически-информированных нейронных сетей применительно к моделированию нефтяного месторождения.

Модели доступные для обучения: UNet, Convolution Autoencoder.

В проекте реализован Physics-Informed loss.

## Требования и установка зависимостей

Проект тестировался в облачном сервисе **ML Space** (Cloud.ru), версия Python 3.9.

Для воспроизведения результатов выполните следующие шаги.

1. Склонируйте данный репозиторий. В терминале перейдите в папку с проектом.
2. Активируйте базовое окружение Anaconda командой `source /home/user/conda/bin/activate`.
3. Создайте виртуальное окружение c Python версии 3.9, выполнив команду `conda create -n py39 python=3.9 anaconda`, где `py39` &ndash; имя окружения.
4. Активируйте созданное окружение &ndash; `source activate py39`.
5. Установите зависимости &ndash; `pip install -r requirements.txt`.

## Запуск обучения

Скрипт для запуска обучения - `train.py`.

Вызов справки по аргументам скрипта:
```commandline
python trainer.py --help
```

Пример запуска обучения модели UNet:

```commandline
 python trainer.py data/pressure1_470.npy data/perm1_470.npy expirement01 --model-name UNet
```
