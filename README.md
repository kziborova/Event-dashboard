<h1 align="center">Event Scenario Lab</h1>
<p align="center">Machine Learning · Data Analytics · Interactive Dashboard</p>
<p align="center">Интеллектуальная система прогнозирования посещаемости мероприятий и сценарного анализа.</p>
<p align="center"><img src="plots/stack.svg" alt="Python · Scikit-learn · Streamlit · Pandas · Plotly"></p>
<p align="center"><a href="https://event-scenario-lab.streamlit.app/"><img src="plots/demo.svg" alt="Live Demo — Streamlit App"></a></p>
<p align="center"><a href="event_analysis.ipynb">Jupyter Notebook ↗</a>　·　<a href="data/events_processed.csv">Dataset ↗</a>　·　<a href="streamlit_app.py">Source Code ↗</a></p>

---

## About the project

Система поддержки принятия решений для организаторов мероприятий на основе анализа исторических данных и методов машинного обучения.

Исследование охватывает 15 585 мероприятий платформы Timepad за 2019–2023 годы.

## Key features

**▥　Data Analytics**<br>
Исследовательский анализ данных и интерактивные графики.

**ƒ　Machine Learning**<br>
Прогнозирование посещаемости с помощью градиентного бустинга.

**⇄　What-if Analysis**<br>
Моделирование влияния цены, бюджета и времени проведения.

**▤　Feature Importance**<br>
Анализ значимости входных признаков.

---

## Machine Learning

Источник данных — Timepad API. Для задачи регрессии ноутбук использует [data/events_raw.csv](data/events_raw.csv). В выборке представлены 23 категории мероприятий, модель получает 17 входных признаков. Рекламный бюджет сформирован синтетически, поскольку соответствующие данные отсутствуют в API Timepad.

Целевая переменная `participants` — число участников. В ноутбуке проверены пропуски и дубликаты, исследованы распределения и корреляции. Из времени начала, цены и вместимости получены категориальные признаки. Для города и категории добавлена средняя посещаемость. Категории закодированы с помощью LabelEncoder.

На случайном разбиении 80/20 (`random_state=42`, 3 117 тестовых событий) сравнили три регрессора. Итоговая модель Gradient Boosting Regressor с 200 деревьями глубиной до 6 дала наименьшую MAE. Дашборд загружает сохранённую модель, кодировщики и порядок признаков через joblib.

| Модель | MAE | RMSE | MAPE | R² |
|:--|--:|--:|--:|--:|
| Linear Regression | 9.49 | 15.61 | 25.05% | 0.881 |
| Random Forest | 6.31 | 11.43 | 12.50% | 0.936 |
| **Gradient Boosting** | **4.66** | **8.11** | **9.37%** | **0.968** |

MAE и RMSE измеряются в участниках. Для бустинга абсолютное отклонение прогноза от цели в среднем составило 4,66 человека. Средняя посещаемость города и категории рассчитана до разбиения выборки, поэтому в признаках есть информация о тестовых целях. Результаты относятся к этому учебному эксперименту. Качество на будущих реальных событиях требует отдельной проверки. [Notebook](event_analysis.ipynb)

---

## Quick Start

macOS/Linux, Python 3.12

```bash
git clone https://github.com/kziborova/Event-dashboard.git
cd Event-dashboard
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

В Windows PowerShell создайте и активируйте окружение следующими командами, затем выполните установку и запуск из примера выше.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

[kziborova · GitHub](https://github.com/kziborova)
