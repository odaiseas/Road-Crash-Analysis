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

# Загрузка данных
def load_data(data_dir, sample=True):
    if sample:
        accidents = pd.read_csv(f"{data_dir}/processed/cleaned_accidents.csv")
        participants = pd.read_csv(f"{data_dir}/processed/cleaned_participants.csv")
        vehicles = pd.read_csv(f"{data_dir}/processed/cleaned_vehicles.csv")
    else:
        raise NotImplementedError("Loading full data not implemented yet")
    return accidents, participants, vehicles

# Нормализация названий регионов
def normalize_region_name(name, region_mapping):
    name = name.strip().lower()
    name = re.sub(r'\bобласть\b', 'обл.', name)
    name = re.sub(r'\bавтономный округ\b', 'ао', name)
    name = re.sub(r'\s+', ' ', name)
    return region_mapping.get(name, name)

# Агрегация данных
def aggregate_accidents(data, group_cols, agg_dict):
    return data.groupby(group_cols).agg(agg_dict).reset_index()

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


# Работа с геоданными
def load_and_merge_geodata(geofile_path, accidents_data, region_col="region"):
    geodata = gpd.read_file(geofile_path)
    geodata[region_col] = geodata[region_col].apply(normalize_region_name, args=(region_mapping,))
    merged = geodata.merge(accidents_data, on=region_col, how="left")
    merged["accident_number"].fillna(0, inplace=True)
    return merged.to_crs('EPSG:32646')

# Вспомогательные функции
def days_in_month(row):
    return calendar.monthrange(row["year"], row["month"])[1]

# Словарь для замены названий регионов
region_mapping = {
    'кемеровская область - кузбасс': 'кемеровская обл.',
    'республика адыгея (адыгея)': 'республика адыгея',
    'республика северная осетия-алания': 'республика северная осетия — алания',
    'республика татарстан (татарстан)': 'республика татарстан',
    'ханты-мансийский автономный округ - югра': 'ханты-мансийский ао — югра',
    'чувашская республика - чувашия': 'чувашская республика',
    'еврейская ао': 'еврейская ао',
    'ненецкий ао': 'ненецкий ао',
    'чукотский ао': 'чукотский ао',
    'ямало-ненецкий ао': 'ямало-ненецкий ао',
    'г. москва': 'москва',
    'г. санкт-петербург': 'санкт-петербург',
    'город федерального значения севастополь': 'севастополь'
}
