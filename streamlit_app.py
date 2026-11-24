from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_PROCESSED = ROOT / "data" / "events_processed.csv"
DATA_RAW = ROOT / "data" / "events_raw.csv"
MODEL_PATH = ROOT / "event_model.pkl"
ENCODERS_PATH = ROOT / "label_encoders.pkl"
FEATURE_COLS_PATH = ROOT / "feature_columns.pkl"
PLOTS_DIR = ROOT / "plots"

DAY_NAMES_RU = [
    "Понедельник",
    "Вторник",
    "Среда",
    "Четверг",
    "Пятница",
    "Суббота",
    "Воскресенье",
]

SEASON_RU = {"winter": "Зима", "spring": "Весна", "summer": "Лето", "autumn": "Осень"}

FEATURE_NAMES_RU = {
    "ad_budget": "Рекламный бюджет",
    "category_enc": "Категория (код)",
    "category_popularity": "Популярность категории",
    "city_enc": "Город (код)",
    "city_popularity": "Популярность города",
    "event_size_enc": "Размер мероприятия",
    "holiday": "Праздник",
    "is_free": "Бесплатное",
    "is_weekend": "Выходной",
    "lead_days": "Дней до события",
    "price": "Цена билета",
    "price_category_enc": "Ценовая категория",
    "season_enc": "Сезон (код)",
    "start_dow": "День недели",
    "start_hour": "Час начала",
    "start_month": "Месяц",
    "time_of_day_enc": "Время суток",
}


def season_from_month(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


RU_FIXED_HOLIDAYS = {
    (1, 1),
    (1, 2),
    (1, 3),
    (1, 4),
    (1, 5),
    (1, 6),
    (1, 7),
    (1, 8),
    (2, 23),
    (3, 8),
    (5, 1),
    (5, 9),
    (6, 12),
    (11, 4),
}


def is_ru_holiday(d: dt.date) -> int:
    return int((d.month, d.day) in RU_FIXED_HOLIDAYS)


def price_category(price: float) -> str:
    if price == 0:
        return "бесплатно"
    if price <= 500:
        return "дешево"
    if price <= 1500:
        return "средне"
    return "дорого"


def time_of_day(hour: int) -> str:
    if 6 <= hour < 12:
        return "утро"
    if 12 <= hour < 17:
        return "день"
    if 17 <= hour < 22:
        return "вечер"
    return "ночь"


def event_size(capacity: int) -> str:
    if capacity <= 30:
        return "малое"
    if capacity <= 100:
        return "среднее"
    return "крупное"


def _safe_label_encode(encoder: Any, value: str, *, fallback: str | None = None) -> int:
    classes = set(getattr(encoder, "classes_", []))
    if value in classes:
        return int(encoder.transform([value])[0])
    if fallback is not None and fallback in classes:
        return int(encoder.transform([fallback])[0])
    return int(encoder.transform([sorted(classes)[0]])[0])


@st.cache_data(show_spinner=False)
def load_processed_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PROCESSED)
    df["starts_dt"] = pd.to_datetime(df["starts_at"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_raw_data() -> pd.DataFrame:
    if not DATA_RAW.exists():
        return pd.DataFrame()
    return pd.read_csv(DATA_RAW)


@st.cache_resource(show_spinner=False)
def load_model_bundle() -> tuple[Any, dict[str, Any], list[str]]:
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    feature_cols = joblib.load(FEATURE_COLS_PATH)
    return model, encoders, feature_cols


@st.cache_data(show_spinner=False)
def popularity_maps(df: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    category_pop = df.groupby("category")["participants"].mean().to_dict()
    city_pop = df.groupby("city")["participants"].mean().to_dict()
    return category_pop, city_pop


def build_feature_row(
    *,
    encoders: dict[str, Any],
    feature_cols: list[str],
    category_popularity: dict[str, float],
    city_popularity: dict[str, float],
    city: str,
    category: str,
    season: str,
    price: int,
    ad_budget: int,
    lead_days: int,
    start_month: int,
    start_dow: int,
    start_hour: int,
    is_weekend: int,
    holiday: int,
    capacity: int,
) -> pd.DataFrame:
    is_free = int(price == 0)
    price_cat = price_category(float(price))
    time_day = time_of_day(int(start_hour))
    event_sz = event_size(int(capacity))

    cat_pop = float(category_popularity.get(category, np.mean(list(category_popularity.values()))))
    city_pop = float(city_popularity.get(city, np.mean(list(city_popularity.values()))))

    features = {
        "city_enc": _safe_label_encode(encoders["city"], city, fallback="Другие"),
        "category_enc": _safe_label_encode(encoders["category"], category, fallback="Не указано"),
        "season_enc": _safe_label_encode(encoders["season"], season, fallback="autumn"),
        "price_category_enc": _safe_label_encode(encoders["price_category"], price_cat, fallback="средне"),
        "time_of_day_enc": _safe_label_encode(encoders["time_of_day"], time_day, fallback="день"),
        "event_size_enc": _safe_label_encode(encoders["event_size"], event_sz, fallback="среднее"),
        "start_month": int(start_month),
        "start_dow": int(start_dow),
        "start_hour": int(start_hour),
        "is_weekend": int(is_weekend),
        "holiday": int(holiday),
        "price": int(price),
        "is_free": int(is_free),
        "ad_budget": int(ad_budget),
        "lead_days": int(lead_days),
        "category_popularity": float(cat_pop),
        "city_popularity": float(city_pop),
    }

    return pd.DataFrame([features])[feature_cols]


def predict_participants(model: Any, X: pd.DataFrame) -> int:
    pred = float(model.predict(X)[0])
    return max(1, int(round(pred)))


def show_png_from_notebook() -> None:
    images = [
        ("Распределение посещаемости", PLOTS_DIR / "01_distribution_participants.png"),
        ("Цена vs посещаемость", PLOTS_DIR / "02_price_vs_attendance.png"),
        ("Посещаемость по дням недели", PLOTS_DIR / "03_attendance_by_day.png"),
        ("Категории", PLOTS_DIR / "04_categories.png"),
        ("Сезонность", PLOTS_DIR / "05_seasonality.png"),
        ("Корреляции", PLOTS_DIR / "06_correlation_matrix.png"),
        ("Сравнение моделей", PLOTS_DIR / "07_model_comparison.png"),
        ("Анализ предсказаний", PLOTS_DIR / "08_predictions_analysis.png"),
        ("Важность признаков", PLOTS_DIR / "09_feature_importance.png"),
    ]
    cols = st.columns(2)
    for i, (title, path) in enumerate(images):
        with cols[i % 2]:
            st.caption(title)
            if path.exists():
                st.image(str(path), use_column_width=True)
            else:
                st.info(f"Файл не найден: {path.name}")


def _plot_distribution(df: pd.DataFrame) -> None:
    c1, c2 = st.columns([2, 1])
    with c1:
        fig = px.histogram(df, x="participants", nbins=50, title="Распределение посещаемости")
        fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=420)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.box(df, y="participants", title="Box-plot")
        fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=420)
        st.plotly_chart(fig, use_container_width=True)


def _plot_price_vs_attendance(df: pd.DataFrame) -> None:
    c1, c2 = st.columns(2)
    with c1:
        d = df[["price", "participants"]].dropna()
        fig = px.scatter(d, x="price", y="participants", opacity=0.35, title="Цена → посещаемость")
        if len(d) >= 2 and d["price"].nunique() >= 2:
            z = np.polyfit(d["price"].to_numpy(), d["participants"].to_numpy(), 1)
            x_line = np.linspace(float(d["price"].min()), float(d["price"].max()), 100)
            y_line = z[0] * x_line + z[1]
            fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", name="Линия тренда"))
        fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=420)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        free_avg = float(df.loc[df["is_free"] == 1, "participants"].mean())
        paid_avg = float(df.loc[df["is_free"] == 0, "participants"].mean())
        fig = px.bar(
            x=["Бесплатные", "Платные"],
            y=[free_avg, paid_avg],
            title="Средняя посещаемость: бесплатные vs платные",
            text=[f"{free_avg:.0f}", f"{paid_avg:.0f}"],
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=420, yaxis_title="Средняя посещаемость")
        st.plotly_chart(fig, use_container_width=True)


def _plot_by_weekday(df: pd.DataFrame) -> None:
    attendance_by_day = df.groupby("start_dow")["participants"].mean().reindex(range(7))
    d = pd.DataFrame({"day": DAY_NAMES_RU, "avg": attendance_by_day.values})
    d["type"] = ["Будни"] * 5 + ["Выходные"] * 2
    fig = px.bar(d, x="day", y="avg", color="type", title="Средняя посещаемость по дням недели", text_auto=".0f")
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=420, xaxis_title="", yaxis_title="Средняя посещаемость")
    st.plotly_chart(fig, use_container_width=True)


def _plot_categories(df: pd.DataFrame) -> None:
    top_n = 10
    category_counts = df["category"].value_counts().head(top_n)
    category_attendance = df.groupby("category")["participants"].mean().sort_values(ascending=False).head(top_n)

    c1, c2 = st.columns([1, 1.3])
    with c1:
        fig = px.pie(
            names=category_counts.index,
            values=category_counts.values,
            title=f"Распределение мероприятий по категориям (Топ-{top_n})",
            hole=0.45,
        )
        fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=420, legend_title_text="Категория")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        d = category_attendance.reset_index().rename(columns={"category": "Категория", "participants": "Средняя посещаемость"})
        fig = px.bar(
            d,
            y="Категория",
            x="Средняя посещаемость",
            orientation="h",
            title=f"Средняя посещаемость по категориям (Топ-{top_n})",
            text_auto=".0f",
        )
        fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=420)
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)


def _plot_seasonality(df: pd.DataFrame) -> None:
    month_names = ["Янв", "Фев", "Мар", "Апр", "Май", "Июн", "Июл", "Авг", "Сен", "Окт", "Ноя", "Дек"]
    attendance_by_month = df.groupby("start_month")["participants"].mean().reindex(range(1, 13))
    d_month = pd.DataFrame({"month": month_names, "avg": attendance_by_month.values})

    df2 = df.copy()
    df2["season_ru"] = df2["season"].map(SEASON_RU).fillna(df2["season"])
    season_order = ["Зима", "Весна", "Лето", "Осень"]
    attendance_by_season = df2.groupby("season_ru")["participants"].mean().reindex(season_order)
    d_season = pd.DataFrame({"season": season_order, "avg": attendance_by_season.values})

    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(d_month, x="month", y="avg", markers=True, title="Сезонность: посещаемость по месяцам")
        fig.add_hline(y=float(attendance_by_month.mean()), line_dash="dash", annotation_text="Среднее", opacity=0.7)
        fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=420, xaxis_title="", yaxis_title="Средняя посещаемость")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(d_season, x="season", y="avg", title="Посещаемость по сезонам", text_auto=".0f")
        fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=420, xaxis_title="", yaxis_title="Средняя посещаемость")
        st.plotly_chart(fig, use_container_width=True)


def _plot_correlation(df: pd.DataFrame) -> None:
    numeric_cols = ["price", "ad_budget", "capacity", "lead_days", "is_weekend", "holiday", "is_free", "participants"]
    corr = df[numeric_cols].corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Матрица корреляций",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=560)
    st.plotly_chart(fig, use_container_width=True)


def page_overview(df: pd.DataFrame) -> None:
    st.title("Система анализа сценариев посещаемости мероприятий")
    st.caption("Дашборд для организаторов: анализ исторических данных + прогноз посещаемости + what-if сценарии.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Мероприятий", f"{len(df):,}".replace(",", " "))
    c2.metric("Категорий", int(df["category"].nunique()))
    c3.metric("Городов", int(df["city"].nunique()))
    c4.metric("Медиана посещаемости", int(df["participants"].median()))

    st.divider()

    c1, c2, c3 = st.columns(3)
    with c1:
        best_day_idx = int(df.groupby("start_dow")["participants"].mean().idxmax())
        st.metric("Лучший день недели", DAY_NAMES_RU[best_day_idx])
    with c2:
        free_avg = float(df.loc[df["is_free"] == 1, "participants"].mean())
        paid_avg = float(df.loc[df["is_free"] == 0, "participants"].mean())
        st.metric("Бесплатные vs платные", f"{free_avg:.0f} / {paid_avg:.0f}")
    with c3:
        best_season = (
            df.assign(season_ru=df["season"].map(SEASON_RU).fillna(df["season"]))
            .groupby("season_ru")["participants"]
            .mean()
            .idxmax()
        )
        st.metric("Лучший сезон", str(best_season))

    st.divider()
    tabs = st.tabs(["Посещаемость", "Цена", "День недели"])
    with tabs[0]:
        _plot_distribution(df)
    with tabs[1]:
        _plot_price_vs_attendance(df)
    with tabs[2]:
        _plot_by_weekday(df)


def page_analytics(df: pd.DataFrame) -> None:
    st.title("Аналитика данных")

    with st.expander("Фильтры", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            cities = sorted(df["city"].dropna().unique().tolist())
            city_sel = st.multiselect("Город", options=cities, default=[])
        with c2:
            categories = sorted(df["category"].dropna().unique().tolist())
            category_sel = st.multiselect("Категория", options=categories, default=[])
        with c3:
            years = sorted(df["start_year"].dropna().unique().astype(int).tolist())
            if years:
                year_min, year_max = min(years), max(years)
                year_range = st.slider("Год", min_value=year_min, max_value=year_max, value=(year_min, year_max))
            else:
                year_range = None

    df_view = df.copy()
    if city_sel:
        df_view = df_view[df_view["city"].isin(city_sel)]
    if category_sel:
        df_view = df_view[df_view["category"].isin(category_sel)]
    if year_range is not None:
        df_view = df_view[(df_view["start_year"] >= year_range[0]) & (df_view["start_year"] <= year_range[1])]

    st.caption(f"Записей после фильтрации: {len(df_view):,}".replace(",", " "))

    show_png = st.toggle("Показать графики как в ноутбуке (PNG)", value=False)
    if show_png:
        show_png_from_notebook()
        return

    tabs = st.tabs(["Посещаемость", "Категории", "Сезонность", "Корреляции"])
    with tabs[0]:
        _plot_distribution(df_view)
        _plot_price_vs_attendance(df_view)
        _plot_by_weekday(df_view)
    with tabs[1]:
        _plot_categories(df_view)
    with tabs[2]:
        _plot_seasonality(df_view)
    with tabs[3]:
        _plot_correlation(df_view)


def page_prediction(df: pd.DataFrame, model: Any | None, encoders: dict[str, Any] | None, feature_cols: list[str] | None) -> None:
    st.title("Прогноз посещаемости и what‑if сценарии")

    if model is None or encoders is None or feature_cols is None:
        st.error(
            "Модель не загрузилась. Проверьте зависимости (scikit-learn должен совпадать с версией при сохранении модели) "
            "и наличие файлов `event_model.pkl`, `label_encoders.pkl`, `feature_columns.pkl`."
        )
        return

    category_pop, city_pop = popularity_maps(df)

    cities = list(getattr(encoders["city"], "classes_", []))
    categories = list(getattr(encoders["category"], "classes_", []))

    default_city = "Москва" if "Москва" in cities else (cities[0] if cities else "")
    default_category = "Концерты" if "Концерты" in categories else (categories[0] if categories else "")

    st.subheader("Параметры будущего мероприятия")

    today = dt.date.today()
    c1, c2, c3 = st.columns(3)
    with c1:
        city = st.selectbox("Город", options=cities, index=(cities.index(default_city) if default_city in cities else 0))
        category = st.selectbox(
            "Категория",
            options=categories,
            index=(categories.index(default_category) if default_category in categories else 0),
        )
    with c2:
        event_date = st.date_input("Дата", value=today + dt.timedelta(days=21), min_value=today)
        start_hour = st.slider("Час начала", min_value=0, max_value=23, value=19, step=1)
    with c3:
        max_price = int(max(5000, df["price"].max()))
        price = st.slider("Цена билета (₽)", min_value=0, max_value=max_price, value=1000, step=50)
        max_budget = int(max(80000, df["ad_budget"].max()))
        ad_budget = st.slider("Рекламный бюджет (₽)", min_value=0, max_value=max_budget, value=30000, step=500)

    c1, c2, c3 = st.columns(3)
    with c1:
        capacity = st.slider("Вместимость (чел.)", min_value=10, max_value=int(max(800, df["capacity"].max())), value=60, step=5)
    with c2:
        lead_default = max(1, int((event_date - today).days))
        lead_days = st.slider("Дней до события", min_value=1, max_value=120, value=min(120, lead_default), step=1)
    with c3:
        holiday_auto = is_ru_holiday(event_date)
        holiday = st.checkbox("Праздник", value=bool(holiday_auto), help="Авто-определение по фиксированным датам РФ (упрощённо).")

    start_month = int(event_date.month)
    start_dow = int(event_date.weekday())
    season = season_from_month(start_month)
    is_weekend = int(start_dow >= 5)

    X_new = build_feature_row(
        encoders=encoders,
        feature_cols=feature_cols,
        category_popularity=category_pop,
        city_popularity=city_pop,
        city=city,
        category=category,
        season=season,
        price=price,
        ad_budget=ad_budget,
        lead_days=lead_days,
        start_month=start_month,
        start_dow=start_dow,
        start_hour=start_hour,
        is_weekend=is_weekend,
        holiday=int(holiday),
        capacity=capacity,
    )

    pred = predict_participants(model, X_new)

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Прогноз посещаемости", f"{pred} чел.")
    c2.metric("Сезон", SEASON_RU.get(season, season))
    c3.metric("День недели", DAY_NAMES_RU[start_dow])
    c4.metric("Формат", "Бесплатное" if price == 0 else "Платное")

    with st.expander("Показать признаки, которые подаются в модель", expanded=False):
        st.dataframe(X_new, use_container_width=True)

    st.subheader("Сценарный анализ (what‑if)")
    param = st.selectbox("Параметр для экспериментов", options=["Цена", "Рекламный бюджет", "Дней до события", "День недели"])

    base_kwargs = dict(
        city=city,
        category=category,
        season=season,
        price=price,
        ad_budget=ad_budget,
        lead_days=lead_days,
        start_month=start_month,
        start_dow=start_dow,
        start_hour=start_hour,
        is_weekend=is_weekend,
        holiday=int(holiday),
        capacity=capacity,
    )

    if param == "Цена":
        n = st.slider("Точек на графике", 10, 60, 25)
        xs = np.linspace(0, max_price, n).round().astype(int)
        ys = []
        for x in xs:
            X_tmp = build_feature_row(
                encoders=encoders,
                feature_cols=feature_cols,
                category_popularity=category_pop,
                city_popularity=city_pop,
                **(base_kwargs | {"price": int(x)}),
            )
            ys.append(predict_participants(model, X_tmp))
        fig = px.line(x=xs, y=ys, markers=False, title="Что если изменить цену?", labels={"x": "Цена (₽)", "y": "Прогноз (чел.)"})
        fig.add_vline(x=price, line_dash="dash", annotation_text="Текущая", opacity=0.7)
        st.plotly_chart(fig, use_container_width=True)
        best_idx = int(np.argmax(ys))
        st.caption(f"Лучшее по модели в диапазоне: цена {xs[best_idx]} ₽ → {ys[best_idx]} чел.")

    if param == "Рекламный бюджет":
        n = st.slider("Точек на графике", 10, 60, 25)
        xs = np.linspace(0, max_budget, n).round().astype(int)
        ys = []
        for x in xs:
            X_tmp = build_feature_row(
                encoders=encoders,
                feature_cols=feature_cols,
                category_popularity=category_pop,
                city_popularity=city_pop,
                **(base_kwargs | {"ad_budget": int(x)}),
            )
            ys.append(predict_participants(model, X_tmp))
        fig = px.line(
            x=xs,
            y=ys,
            markers=False,
            title="Что если изменить рекламный бюджет?",
            labels={"x": "Бюджет (₽)", "y": "Прогноз (чел.)"},
        )
        fig.add_vline(x=ad_budget, line_dash="dash", annotation_text="Текущий", opacity=0.7)
        st.plotly_chart(fig, use_container_width=True)
        best_idx = int(np.argmax(ys))
        st.caption(f"Лучшее по модели в диапазоне: бюджет {xs[best_idx]} ₽ → {ys[best_idx]} чел.")

    if param == "Дней до события":
        n = st.slider("Точек на графике", 10, 60, 25)
        xs = np.linspace(1, 120, n).round().astype(int)
        ys = []
        for x in xs:
            X_tmp = build_feature_row(
                encoders=encoders,
                feature_cols=feature_cols,
                category_popularity=category_pop,
                city_popularity=city_pop,
                **(base_kwargs | {"lead_days": int(x)}),
            )
            ys.append(predict_participants(model, X_tmp))
        fig = px.line(
            x=xs,
            y=ys,
            markers=False,
            title="Что если начать продвижение раньше/позже? (lead_days)",
            labels={"x": "Дней до события", "y": "Прогноз (чел.)"},
        )
        fig.add_vline(x=lead_days, line_dash="dash", annotation_text="Текущие", opacity=0.7)
        st.plotly_chart(fig, use_container_width=True)
        best_idx = int(np.argmax(ys))
        st.caption(f"Лучшее по модели в диапазоне: lead_days {xs[best_idx]} → {ys[best_idx]} чел.")

    if param == "День недели":
        xs = np.arange(7, dtype=int)
        ys = []
        for dow in xs:
            X_tmp = build_feature_row(
                encoders=encoders,
                feature_cols=feature_cols,
                category_popularity=category_pop,
                city_popularity=city_pop,
                **(base_kwargs | {"start_dow": int(dow), "is_weekend": int(dow >= 5)}),
            )
            ys.append(predict_participants(model, X_tmp))
        d = pd.DataFrame({"day": DAY_NAMES_RU, "pred": ys, "type": ["Будни"] * 5 + ["Выходные"] * 2})
        fig = px.bar(d, x="day", y="pred", color="type", title="Что если выбрать другой день недели?", text_auto=True)
        fig.add_hline(y=pred, line_dash="dash", annotation_text="Текущий прогноз", opacity=0.7)
        st.plotly_chart(fig, use_container_width=True)
        best_day = int(xs[int(np.argmax(ys))])
        st.caption(f"Лучшее по модели: {DAY_NAMES_RU[best_day]} → {max(ys)} чел.")


def page_model(df: pd.DataFrame, model: Any | None, feature_cols: list[str] | None) -> None:
    st.title("Модель и факторы влияния")

    if model is None or feature_cols is None:
        st.info("Модель не загружена — показываю графики из ноутбука.")
        show_png_from_notebook()
        return

    st.caption(f"Алгоритм: `{type(model).__name__}` • Признаков: {len(feature_cols)}")

    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=float)
        d = pd.DataFrame({"feature": feature_cols, "importance": imp})
        d["feature_ru"] = d["feature"].map(FEATURE_NAMES_RU).fillna(d["feature"])
        d = d.sort_values("importance", ascending=False)

        fig = px.bar(
            d.head(15),
            x="importance",
            y="feature_ru",
            orientation="h",
            title="Важность признаков (Топ-15)",
            text_auto=".2%",
        )
        fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=560, xaxis_title="Важность", yaxis_title="")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Графики из ноутбука")
    show_png_from_notebook()


def main() -> None:
    st.set_page_config(page_title="Event Attendance Optimizer", layout="wide")

    try:
        df = load_processed_data()
    except Exception as e:
        st.error(f"Не удалось загрузить датасет: {e}")
        return

    try:
        model, encoders, feature_cols = load_model_bundle()
    except Exception:
        model, encoders, feature_cols = None, None, None

    with st.sidebar:
        st.header("Навигация")
        page = st.radio("Раздел", options=["Обзор", "Аналитика", "Прогноз", "Модель"], index=0)
        st.divider()
        st.caption("Файлы:")
        st.caption(f"- Данные: `{DATA_PROCESSED.name}`")
        st.caption(f"- Модель: `{MODEL_PATH.name}`")

    if page == "Обзор":
        page_overview(df)
        return
    if page == "Аналитика":
        page_analytics(df)
        return
    if page == "Прогноз":
        page_prediction(df, model, encoders, feature_cols)
        return
    if page == "Модель":
        page_model(df, model, feature_cols)
        return


if __name__ == "__main__":
    main()

