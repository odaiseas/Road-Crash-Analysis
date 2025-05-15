# -*- coding: utf-8 -*-
"""
Created on Tue May  6 12:12:10 2025

@author: Admin
"""
# scripts/utils.py
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import re
import calendar
from typing import Dict, Tuple

# Загрузка данных
def load_data(data_dir, sample=True):
    if sample:
        accidents = pd.read_csv(f"{data_dir}/processed/cleaned_accidents.csv")
        participants = pd.read_csv(f"{data_dir}/processed/cleaned_participants.csv")
        vehicles = pd.read_csv(f"{data_dir}/processed/cleaned_vehicles.csv")
    else:
        raise NotImplementedError("Loading full data not implemented yet")
    return accidents, participants, vehicles

# Агрегация данных
def aggregate_accidents(data, group_cols, agg_dict):
    return data.groupby(group_cols).agg(agg_dict).reset_index()
 
 # Агрегация данных по регионам
def compute_accident_count(
    df: pd.DataFrame,
    group_col: str = 'region',
    id_col: str = 'id'
) -> pd.DataFrame:
    """
    Возвращает DataFrame с колонками:
    [group_col, 'accident_number'] — число ДТП в каждой группе.
    """
    result = (
        df
        .copy()
        .assign(**{group_col: df[group_col].apply(normalize_region_name)})
        .groupby(group_col)
        .agg(accident_number=(id_col, 'size'))
        .reset_index()
    )
    return result

# Визуализация: линейный график
def plot_line_chart(x, y, x_labels, title, xlabel, ylabel, figsize=(16, 6)):
    plt.figure(figsize=figsize)
    plt.plot(range(len(x_labels)), y, marker="o", linestyle="-", color="blue")
    plt.xticks(ticks=range(len(x_labels)), labels=x_labels, rotation=45, ha="right", fontsize=10)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.title(title, fontsize=14)
    plt.show()

# Визуализация: столбчатая диаграмма
def plot_bar_chart(x, y, title, xlabel, ylabel, figsize=(16, 6), rotation=45):
    plt.figure(figsize=figsize)
    plt.bar(x, y, color="blue", alpha=0.7)
    plt.xticks(rotation=rotation, ha="right", fontsize=10)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(False)
    plt.title(title, fontsize=14)
    plt.show()
    
# Визуализация: интерактивный линейный график
def plotly_line_chart(x, y, x_labels=None,
                      title='', xlabel='', ylabel='',
                      width=800, height=400):
    """
    Построить интерактивный линейный график через Plotly Express.
    x — последовательность значений по оси X;
    y — последовательность значений по оси Y;
    x_labels — подписи для xticks (список строк) или None;
    """
    fig = px.line(
        x=list(range(len(x) if x_labels else x)),
        y=y,
        markers=True,
        title=title
    )
    fig.update_layout(
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        width=width,
        height=height,
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(x_labels))) if x_labels else None,
            ticktext=x_labels
        ),
        template='simple_white'
    )
    fig.show()
   
# Визуализация: интерактивная столбчатая диаграмма
def plotly_bar_chart(
    x, y,
    x_labels=None,
    title: str = '',
    xlabel: str = '',
    ylabel: str = '',
    width: int = 800,
    height: int = 400,
    tickangle: int = None,
    margin: dict = None
):
    """
    Интерактивная столбчатая диаграмма через Plotly Express.
    x        — список категорий (например, регионы);
    y        — числовые значения для высоты столбцов;
    x_labels — подписи для оси X (список строк);
    tickangle— угол поворота подписей по оси X (в градусах), опц.;
    margin   — отступы фигуры dict(l, r, t, b), опц.
    """
    import plotly.express as px

    if margin is None:
        margin = dict(l=0, r=0, t=50, b=150)

    idx = list(range(len(x)))
    fig = px.bar(
        x=idx,
        y=y,
        text=y,
        title=title
    )
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        width=width,
        height=height,
        template='simple_white',
        margin=margin,
        xaxis=dict(
            tickmode='array',
            tickvals=idx,
            ticktext=x_labels
        ),
        yaxis=dict(showgrid=True, gridcolor='lightgrey')
    )
    if tickangle is not None:
        fig.update_layout(xaxis_tickangle=tickangle)
    fig.show()
    
# Тяжесть ДТП по годам
def compute_death_rate(
    df: pd.DataFrame,
    group_col: str = 'year',
    value_col: str = 'dead_count'
) -> pd.DataFrame:
    """
    Группирует df по столбцу group_col и возвращает DataFrame с тремя колонками:
    - group_col (например, год)
    - 'accident_count'    — число ДТП (size)
    - 'death_count'       — сумма погибших (sum)
    - 'deaths_per_accident' — отношение погибших к числу ДТП
    """
    result = (
        df
        .groupby(group_col)
        .agg(
            accident_count=(value_col, 'size'),
            death_count   =(value_col, 'sum')
        )
        .reset_index()
    )
    result['deaths_per_accident'] = (
        result['death_count'] / result['accident_count']
    )
    return result

# Обобщённая агрегация метрик по группам
def aggregate_metrics(
    df: pd.DataFrame,
    group_col: str,
    metrics: Dict[str, Tuple[str, str]]
) -> pd.DataFrame:
    """
    Группирует df по group_col и возвращает DataFrame с колонками:
      - group_col
      - для каждого new_col в metrics: new_col = aggfunc(df[src_col])
    """
    agg_kwargs = {
        new_col: (src_col, func_name)
        for new_col, (src_col, func_name) in metrics.items()
    }
    result = df.groupby(group_col).agg(**agg_kwargs).reset_index()
    return result

# Расчёт относительного показателя (rate)
def compute_rate(
    df: pd.DataFrame,
    numerator: str,
    denominator: str,
    new_col: str,
    multiplier: float = 1.0
) -> pd.DataFrame:
    """
    Добавляет в df колонку new_col = (df[numerator] / df[denominator]) * multiplier.
    """
    df = df.copy()
    df[new_col] = (df[numerator] / df[denominator]) * multiplier
    return df

# Построение столбчатых диаграмм
def plot_bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str = '',
    xlabel: str = '',
    ylabel: str = '',
    figsize: Tuple[int, int] = (16, 6),
    color: str = 'blue',
    alpha: float = 0.7,
    rotate: float = 45
):
    """
    Построение однородных столбчатых диаграмм через Matplotlib.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(df[x], df[y], color=color, alpha=alpha)
    ax.grid(False)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=rotate, ha='right')
    plt.tight_layout()
    plt.show()

# Работа с геоданными
region_mapping = {
    'кемеровская область - кузбасс': 'кемеровская обл.',
    'республика адыгея (адыгея)': 'республика адыгея',
    # … остальные мэппинги …
}

def normalize_region_name(name: str, mapping: dict = region_mapping) -> str:
    """
    Приводит name к нижнему регистру, заменяет 
    'область'→'обл.', 'автономный округ'→'ао' 
    и затем ищет в mapping.
    """
    n = name.strip().lower()
    n = re.sub(r'\bобласть\b', 'обл.', n)
    n = re.sub(r'\bавтономный округ\b', 'ао', n)
    return mapping.get(n, n)

def load_and_merge_geodata(
    geofile_path: str,
    accidents_data: pd.DataFrame,
    region_col: str = "region"
) -> gpd.GeoDataFrame:
    """
    1) Загружает GeoJSON по относительному пути geofile_path.
    2) Нормализует колонку region_col в обоих DataFrame.
    3) Мёрджит по region_col и заполняет NaN в accident_number нулями.
    4) Переводит в CRS EPSG:32646.
    """
    gdf = gpd.read_file(geofile_path)
    # нормализуем названия в геоданных
    gdf[region_col] = gdf[region_col].apply(normalize_region_name)
    # нормализуем в таблице ДТП
    accidents_data = (
        accidents_data
        .copy()
        .assign(**{region_col: accidents_data[region_col].apply(normalize_region_name)})
    )
    merged = gdf.merge(accidents_data, on=region_col, how='left')
    merged['accident_number'] = merged['accident_number'].fillna(0)
    return merged.to_crs('EPSG:32646')
    
# Загрузка полного набора данных в SQLite
def load_full_data_to_sqlite(acc_url, part_url, veh_url, db_path):
    """Скачивает CSV с Яндекс.Диска, сохраняет в sqlite и возвращает соединение."""
    import sqlite3, pandas as pd, requests
    conn = sqlite3.connect(db_path)
    for public_url, table in [(acc_url, 'accidents'),
                              (part_url,'participants'),
                              (veh_url, 'vehicles')]:
        href = requests.get(
            "https://cloud-api.yandex.net/v1/disk/public/resources/download",
            params={"public_key": public_url}
        ).json()["href"]
        pd.read_csv(href, sep=';') \
          .to_sql(table, conn, if_exists='replace', index=False)
    return conn
    
# Универсальный запуск SQL-запроса
def run_query(conn, sql: str) -> pd.DataFrame:
    """Обёртка над pd.read_sql_query с единым логом."""
    df = pd.read_sql_query(sql, conn)
    print(f"Query returned {len(df)} rows")
    return df
    
# SQL-запрос с объединением таблиц ДТП, участников и ТС
SQL_ACCIDENT_FEATURES = """
WITH drivers AS (
    SELECT *
    FROM participants
    WHERE role = 'Водитель'
),

violating_drivers AS (
    SELECT *
    FROM drivers
    WHERE violations IS NOT NULL AND TRIM(violations) != ''
),

selected_drivers AS (
    SELECT accident_id, participant_id, gender, years_of_driving_experience,
           CASE
               WHEN COUNT(*) = 1 THEN 'single_violator'
               ELSE 'multiple_violators'
           END AS violator_case
    FROM violating_drivers
    GROUP BY accident_id
),

fallback_drivers AS (
    SELECT accident_id, participant_id, gender, years_of_driving_experience,
           'no_violator' AS violator_case
    FROM drivers
    WHERE accident_id NOT IN (SELECT DISTINCT accident_id FROM violating_drivers)
),

final_drivers AS (
    SELECT * FROM selected_drivers
    UNION ALL
    SELECT * FROM fallback_drivers
),

driver_vehicles AS (
    SELECT d.accident_id, p.vehicle_id
    FROM final_drivers d
    JOIN participants p ON d.participant_id = p.participant_id
    WHERE p.vehicle_id IS NOT NULL
),

vehicle_years AS (
    SELECT dv.accident_id,
           AVG(v.year) AS avg_vehicle_year
    FROM driver_vehicles dv
    JOIN vehicles v ON dv.vehicle_id = v.vehicle_id
    JOIN accidents a ON a.id = dv.accident_id
    GROUP BY dv.accident_id
),

driver_aggregates AS (
    SELECT accident_id,
           COUNT(*) AS driver_count,
           AVG(years_of_driving_experience) AS avg_experience,
           CASE
               WHEN SUM(CASE WHEN gender = 'Женский' THEN 1 ELSE 0 END) = 0 THEN 'all_male'
               WHEN SUM(CASE WHEN gender = 'Мужской' THEN 1 ELSE 0 END) = 0 THEN 'all_female'
               ELSE 'mixed'
           END AS drivers_gender
    FROM final_drivers
    GROUP BY accident_id
),

pedestrian_flag AS (
    SELECT accident_id,
           MAX(CASE WHEN role = 'Пешеход' AND violations IS NOT NULL AND TRIM(violations) != '' THEN 1 ELSE 0 END) AS has_violating_pedestrian
    FROM participants
    GROUP BY accident_id
)

SELECT 
    da.accident_id,
    a.category,
    a.datetime,
    a.light,
    a.weather,
    a.participants_count,
    a.dead_count,
    a.injured_count,
    da.driver_count,
    da.avg_experience,
    da.drivers_gender,
    va.avg_vehicle_year,
    pf.has_violating_pedestrian
FROM driver_aggregates AS da
JOIN accidents AS a ON da.accident_id = a.id  -- Добавляем JOIN с accidents
LEFT JOIN vehicle_years AS va ON da.accident_id = va.accident_id
LEFT JOIN pedestrian_flag AS pf ON da.accident_id = pf.accident_id;
"""  
  
def fetch_accident_features(conn, sql=SQL_ACCIDENT_FEATURES):
    """Выполняет заранее определённый SQL и возвращает DataFrame."""
    import pandas as pd
    return pd.read_sql_query(sql, conn)

# Переменные для регрессионных моделей
PREDICTORS = [
    'remainder__adverse_weather',
    'remainder__is_weekend',
    'remainder__has_violating_pedestrian',
    'remainder__drivers_gender_female',
    'cat__light_Светлое время суток',
    'cat__light_В темное время суток, освещение отсутствует',
    'cat__light_Сумерки',
    'cat__light_В темное время суток, освещение включено',
    'cat__category_Столкновение',
    'cat__category_Наезд на пешехода',
    'cat__category_Наезд',
    'cat__category_Опрокидывание',
    'cat__category_Съезд с дороги',
    'remainder__sin_time',
    'remainder__cos_time',
    'remainder__avg_experience',
    'remainder__avg_vehicle_age'
]

TARGET    = 'remainder__dead_count'
EXPOSURE  = 'remainder__participants_count'
 
# Сплит и подготовка для Statsmodels
def prepare_train_test(df, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split
    import statsmodels.api as sm

    X = df[PREDICTORS].astype(float)
    y = df[TARGET].astype(float)
    exp = df[EXPOSURE].astype(float)

    X_train, X_test, y_train, y_test, exp_train, exp_test = train_test_split(
        X, y, exp, test_size=test_size, random_state=random_state
    )
    # добавляем константу
    X_train_sm = sm.add_constant(X_train)
    X_test_sm  = sm.add_constant(X_test)
    return X_train_sm, X_test_sm, y_train, y_test, exp_train, exp_test 

# Обучение регрессионных моделей
# utils.py
def fit_negative_binomial(X, y, exposure, method='newton', maxiter=100):
    import numpy as np
    import statsmodels.api as sm

    model = sm.NegativeBinomial(endog=y, exog=X, offset=np.log(exposure))
    return model.fit(method=method, maxiter=maxiter, disp=False)

def fit_poisson(X, y, exposure):
    import numpy as np
    import statsmodels.api as sm

    model = sm.GLM(
        endog=y,
        exog=X,
        family=sm.families.Poisson(),
        offset=np.log(exposure)
    )
    return model.fit()

# Оценка качества регрессионных моделей
def evaluate_mse(result, X_test, y_test, exposure_test):
    from sklearn.metrics import mean_squared_error
    import numpy as np

    y_pred = result.predict(X_test, offset=np.log(exposure_test))
    mask   = ~np.isnan(y_pred)
    return mean_squared_error(y_test[mask], y_pred[mask])
    
# Таблица коэффициентов регрессионных моделей
def coef_table(result, drop_vars=None, sort_by='P-value'):
    import pandas as pd

    df = pd.DataFrame({
        'Variable': result.params.index,
        'Coefficient': result.params.values,
        'P-value': result.pvalues.values
    }).round({'Coefficient': 3, 'P-value': 4})

    df['Significant (p<0.05)'] = df['P-value'] < 0.05
    # вынести const (и любые drop_vars) вниз и сортировать
    main = df[~df['Variable'].isin(drop_vars or [])].sort_values(by=sort_by)
    tail = df[df['Variable'].isin(drop_vars or [])]
    return pd.concat([main, tail]).reset_index(drop=True)

# Визуализация коэффициентов регрессионных моделей
def plot_significant_coefs(coef_df, output_path, threshold=0.05, exclude=['const','alpha']):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    sig = coef_df[
        (coef_df['Significant (p<0.05)']) &
        (~coef_df['Variable'].isin(exclude))
    ].copy()
    sig = sig.reindex(sig['Coefficient'].sort_values(ascending=False).index)
    sig['Effect'] = sig['Coefficient'].apply(lambda x: 'Increase' if x>0 else 'Decrease')

    plt.figure(figsize=(10, 6))
    sns.barplot(
        y='Variable', x='Coefficient',
        data=sig, orient='h',
        hue='Effect', dodge=False, legend=False
    )
    plt.axvline(0, color='gray', linestyle='--')
    plt.title('Significant coefficients (p < {})'.format(threshold))
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()

 
# Сохранение графиков
def save_png(fig, path: str, dpi=300):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)

def save_html(fig, path: str):
    fig.write_html(path)