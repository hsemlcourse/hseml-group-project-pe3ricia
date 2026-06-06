[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)
# ML Project — AML Risk Score Classification

**Студент:** Коновченко Петр Михайлович  
**Группа:** БИВ236


## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Быстрый старт](#быстрый-старт)
4. [Запуск через Docker](#запуск-через-docker)
5. [Данные](#данные)
6. [Результаты](#результаты)
7. [Отчёт](#отчёт)


## Описание задачи

**Задача:** многоклассовая классификация (10 классов)

**Датасет:** [Global Black Money Transactions Dataset](https://www.kaggle.com/) — 10 000 транзакций, 14 признаков

**Целевая переменная:** `Money Laundering Risk Score` — уровень риска отмывания денег (1–10)

**Основная метрика:** weighted F1-score


## Структура репозитория

```
.
├── data
│   ├── raw/                        # Исходный CSV (black_money_transactions.csv)
│   └── processed/                  # train/valid/test в форматах .csv и .pkl
├── models/                         # Сохранённые модели (.pkl) и таблица экспериментов
├── notebooks
│   ├── EDA_risk.ipynb              # Разведочный анализ данных
│   ├── baseline_risk.ipynb         # Baseline-модель
│   └── experiments_risk.ipynb      # Эксперименты с моделями
├── presentation/                   # Презентация для защиты
├── report
│   ├── *.png                       # Графики EDA
│   └── report.md                   # Финальный отчёт
├── src
│   ├── config.py                   # Пути, seed, списки признаков
│   ├── preprocessing.py            # Feature engineering + разбивка данных
│   ├── eda.py                      # EDA-графики и статистики
│   ├── baseline.py                 # Baseline LogisticRegression
│   ├── experiments.py              # Grid-search по 4 моделям
│   └── train.py                    # Финальная модель (train+val → test)
├── tests/
│   └── test.py                     # Тесты пайплайна
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml                  # Конфиг ruff
├── requirements.txt
└── README.md
```


## Быстрый старт

```bash
# 1. Клонировать репозиторий
git clone <url>
cd <repo-name>

# 2. Создать виртуальное окружение
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows

# 3. Установить зависимости
pip install -r requirements.txt

# 4. Запустить полный пайплайн
python -m src.preprocessing    # feature engineering + сплит
python -m src.eda              # EDA-графики → report/
python -m src.baseline         # baseline модель
python -m src.experiments      # grid-search экспериментов
python -m src.train            # финальная модель → models/final_model.pkl

# Линтинг
ruff check src/
```


## Запуск через Docker

```bash
# Собрать образ
docker compose build

# Запустить отдельный этап (пример)
docker compose run preprocess
docker compose run train

# Или все этапы последовательно
docker compose run preprocess && \
docker compose run eda && \
docker compose run baseline && \
docker compose run experiments && \
docker compose run train
```


## Данные

- `data/raw/black_money_transactions.csv` — исходный датасет (10 000 строк, 14 признаков)
- `data/processed/` — предобработанные данные: разбивка train/valid/test (70/15/15), сохранена в `.csv` и `.pkl`

Предобработка (`src/preprocessing.py`): извлечение временных признаков из даты, бинарные флаги (`is_illegal`, `is_reported`, `has_tax_haven`), логарифм суммы, группировка редких банков.


## Результаты

| Модель | Val Accuracy | Val F1 (weighted) | Test F1 (weighted) |
|--------|-------------|-------------------|-------------------|
| Baseline (LogReg, LabelEnc) | 0.1027 | 0.0742 | 0.0818 |
| LogisticRegression + OHE | 0.1073 | 0.0704 | 0.0721 |
| KNN + OHE (best grid) | 0.1160 | 0.1157 | 0.1057 |
| GradientBoosting + OHE (best grid) | 0.1107 | 0.1083 | 0.1072 |
| **RandomForest + OHE (best grid)** | **0.1220** | **0.1217** | 0.1063 |
| RandomForest финальный (train+val) | — | — | **0.0994** |

Все модели работают вблизи случайного угадывания (1/10 = 0.10): числовые признаки не коррелируют с таргетом (|r| < 0.03), что характерно для синтетических данных.


## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md)
