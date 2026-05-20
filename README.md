[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)
# ML Project — Laundered Funds Amount Prediction

**Студент:** Коновченко Пётр Михайлович

**Группа:** БИВ 236


## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Запуски](#быстрый-старт)
4. [Данные](#данные)
5. [Результаты](#результаты)
7. [Отчёт](#отчёт)


## Описание задачи

<!-- Кратко опишите задачу: что предсказываем, какой датасет, метрика качества -->

**Задача:** Регрессия. Предсказываем сумму перевода по признакам транзакции

**Датасет:** Global Black Money Transactions Dataset с Kaggle

**Целевая метрика:** RMSE


## Структура репозитория
Опишите структуру проекта, сохранив при этом верхнеуровневые папки. Можно добавить новые при необходимости.
```
.
├── data
│   ├── processed               # Очищенные и обработанные данные
│   │   ├── test.csv
│   │   ├── train.csv
│   │   └── valid.csv
│   └── raw                     # Исходные файлы
│       └── black_money_transactions.csv
├── models                      # Сохранённые модели 
│       └── baseline_linear_regression.pkl
├── notebooks
│   ├── 01_eda.ipynb            # EDA
│   └── 02_baseline.ipynb       # Baseline-модель
├── presentation                # Презентация для защиты
├── report
├── src
│   ├── preprocessing.py        # Предобработка данных
│   └── modeling.py             # Обучение и оценка базовой модели
├── tests
├── requirements.txt
└── README.md
```

## Запуск

Этот блок замените способом запуска вашего сервиса.
```bash
# 1. Клонировать репозиторий
git clone <url>
cd <repo-name>

# 2. Создать виртуальное окружение
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# 3. Установить зависимости
pip install -r requirements.txt
```

## Данные
- `data/raw/` — исходные файлы
- `data/processed/` — предобработанные данные


## Результаты
Здесь коротко выпишите результаты.
| Модель | [Метрика 1] | [Метрика 2] | Примечание |
|--------|-------------|-------------|------------|
| Baseline | — | — | |
| Лучшая модель | — | — | |


## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md)
