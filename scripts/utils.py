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
    
# Сохранение графиков
def save_png(fig, path: str, dpi=300):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)

def save_html(fig, path: str):
    fig.write_html(path)