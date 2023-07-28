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


## Структура проекта

- dataset.py &ndash; код для предобработки данных;
- loss.py &ndash; реализация физически-информированной функции потерь;
- trainer.py &ndash; крипт запуска обучения моделей;
- utils.py &ndash; вспомогательный код;
- resize.py &ndash; код для улучшения разрешения семплов датасета низкого разрешения;
- q_estimate.py &ndash; код для экспериментов по вычислению q по данным;
- numeric_darcy.ipynb &ndash; тетрадка с реализацией численного метода решения уравнения Дарси (работает некорректно).
- data &ndash; папка для датасетов;


## Запуск обучения

Перед запуском обучения моделей, поместите в папку `data` необходимые датасеты. Скрипт для запуска обучения &ndash; `train.py`.

Вызов справки по аргументам скрипта:
```commandline
python trainer.py --help
```

Пример запуска обучения модели UNet:

```commandline
 python trainer.py data/pressure1_470.npy data/perm1_470.npy expirement01 --model-name UNet
```


## Результаты обучения

Результаты обучения доступны через Tensorboard. Логи сохраняются в папаку `results`. Для их просмотра из облачного сервиса ML Space необходимо:

1) Запустить сервис командой `tensorboard --logdir_spec results_all:results/`
2) Пробросить порты в терминале на клиентском компьютере командой `ssh -L 9999:localhost:6006 xxxxxxxxxx.xxxxxxxxxxx@ssh-jupyter.aicloud.sbercloud.ru -p 2222 -i mlspace__private_key.txt`
где команда `xxxxxxxxxx.xxxxxxxxxxx@ssh-jupyter.aicloud.sbercloud.ru -p 2222 -i` генерируется в консоли ML Space, `mlspace__private_key.txt` &ndash; секретный ключ, генерируется там же. 
3) Просмотр логов будет доступен по адресу `http://localhost:9999/`


## Возможные ошибки при запуске

При запуске обучения модели ConvAutoencoder возможна следующая ошибка:

```commandline
RuntimeError: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or 
`at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable 
deterministic behavior in this case, you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or 
CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
```

для ее устранения можно установить переменную среды:

```commandline
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```
