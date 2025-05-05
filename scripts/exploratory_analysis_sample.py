# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 12:38:54 2025

@author: Admin
"""
#%%
"""Краткий анализ выборочных данных, загруженных на гитхаб, для воспроизводимости результатов без необходимости загружать исходные файлы из облака"""
#%% Импорт библиотек и данных
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import geopandas as gpd
import calendar
import locale
import re
sample_accidents = pd.read_csv('C:/Users/admin/Road-Crash-Analysis/data/processed/cleaned_accidents.csv')
sample_participants = pd.read_csv('C:/Users/admin/Road-Crash-Analysis/data/processed/cleaned_participants.csv')
sample_vehicles = pd.read_csv('C:/Users/admin/Road-Crash-Analysis/data/processed/cleaned_vehicles.csv')
#%% Многолетнаяя динамика числа ДТП
# Считаем количество ДТП по месяцам + группировка по годам и месяцам с добавлением суммы числа погибших
monthly_accidents = (
    sample_accidents
    .groupby(["year", "month"])
    .agg(
        accident_number=("dead_count", "size"),  # Количество ДТП
        death_number=("dead_count", "sum")   # Сумма числа погибших
    )
    .reset_index()
)

# Номера месяцев и годы для графика
month_numbers = monthly_accidents["month"].tolist()
years = monthly_accidents["year"].astype(str).tolist()
# Создание меток оси X в формате '2015-1, 2, ..., 12, 2016-1, ...'
x_labels = [
    f"{years[i]}-{month_numbers[i]}" if month_numbers[i] == 1 else f"{month_numbers[i]}"
    for i in range(len(month_numbers))
]

# Построение графика
plt.figure(figsize=(16, 6))
plt.plot(range(len(x_labels)), monthly_accidents["accident_number"], marker="o", linestyle="-", color="blue", label="Количество ДТП")

# Настройка оси X
plt.xticks(ticks=range(len(x_labels)), labels=x_labels, rotation=45, ha="right", fontsize=10)
plt.xlabel("Время (год-месяц)", fontsize=12)
plt.ylabel("Количество ДТП", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)

"""
За рассматриваемый период общее число ДТП постепенно сокращалось, по крайней мере с 2016 года. График несглаженный, что является особенностью работы с небольшой выборкой 
"""

#%% Распределение ДТП по месяцам

# Добавляем число дней в каждом месяце
def days_in_month(row):
    return calendar.monthrange(row["year"], row["month"])[1]

monthly_accidents["days_per_month"] = monthly_accidents.apply(days_in_month, axis=1)

# Рассчитываем среднедневное число ДТП 
monthly_accidents["accidents_per_day"] = (monthly_accidents["accident_number"] / monthly_accidents["days_per_month"])

# Сортируем для удобства
monthly_accidents = monthly_accidents.sort_values(by=["year", "month"]).reset_index(drop=True)

# Добавляем взвешенную сумму "количество_ДТП" для каждого месяца
weighted_data = (
    monthly_accidents.groupby("month", group_keys=False)  # Исключаем группировочные столбцы
    .apply(lambda group: pd.Series({
        "total_accidents": group["accident_number"].sum(),
        "total_days": group["days_per_month"].sum(),
    }, index=["total_accidents", "total_days"]))
    .reset_index()
)

# Рассчитываем среднее ДТП в день
weighted_data["avg_dayly_accident"] = (weighted_data["total_accidents"] / weighted_data["total_days"]).round(3)

# Проверяем результат
print(weighted_data)
#%% Построение графика для распределения ДТП по месяцам
"""
На графике отображены средние значения ДТП в день для каждого месяца за все годы. Значения вычислены с учётом разной длины месяцев.
Летние месяцы и сентябрь демонстрируют более высокие значения, что может быть связано с увеличением числа поездок, отпускным сезоном и ростом интенсивности движения.
Весенние март и февраль характеризуются более низкими значениями, что может быть связано с улучшением дорожных условий после зимы и меньшей интенсивностью движения.
"""
# Словарь с русскими названиями месяцев
month_names_ru = {
    1: "Январь", 2: "Февраль", 3: "Март", 4: "Апрель",
    5: "Май", 6: "Июнь", 7: "Июль", 8: "Август",
    9: "Сентябрь", 10: "Октябрь", 11: "Ноябрь", 12: "Декабрь"
}
# Подготовка данных
labels = weighted_data["month"].tolist()  # Названия месяцев из столбца 'month_name'
values = weighted_data["avg_dayly_accident"].tolist()  # Значения ДТП в день из столбца 'ДТП_в_день'

# Преобразуем числа месяцев в русские названия
labels = [month_names_ru[month] for month in labels]

# Углы для каждого значения
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()

# Закрываем круг (один элемент добавляется в оба массива)
values.append(values[0])  # Добавляем первое значение в конец
angles.append(angles[0])  # Добавляем первый угол в конец

# Убедимся, что размеры совпадают
assert len(values) == len(angles), f"Несоответствие размеров: values={len(values)}, angles={len(angles)}"

# Построение графика
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
ax.plot(angles, values, linewidth=2, linestyle="solid", label="ДТП в день")
ax.fill(angles, values, color="blue", alpha=0.25)

# Установка меток
ax.set_xticks(angles[:-1])  # Углы для меток (без замыкающего)
ax.set_xticklabels(labels, fontsize=10)  # Русские названия месяцев

plt.show()
#%% Распределение ДТП по времени суток
sample_accidents["datetime"] = pd.to_datetime(sample_accidents["datetime"])
sample_accidents["hour"] = sample_accidents["datetime"].dt.hour
hourly_data = sample_accidents.groupby(["hour"]).size().reset_index(name="accident_number")
"""Добавлено в 01_data_cleaning.ipynb, позже убрать"""
#Число ДТП по часам
x_labels = hourly_data["hour"]
"""ночью происходит существенно меньше ДТП, чем в другое время суток, с 5 утра их число начинает быстро расти достигая локального максимума в 8 часов утра, то есть в утренний час пик, несколько снижается к 9 часам утра и далее непрерывно растет вплоть до абсолютного максимума в вечерний час пик в 17-19 часов, после чего быстро снижается"""
# Построение графика
plt.figure(figsize=(16, 6))
plt.plot(range(len(x_labels)), hourly_data["accident_number"], marker="o", linestyle="-", color="blue", label="Количество ДТП")
# Настройка оси X
plt.xticks(ticks=range(len(x_labels)), labels=x_labels, rotation=45, ha="right", fontsize=10)
plt.xlabel("Время (час)", fontsize=12)
plt.ylabel("Количество ДТП", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()
#%% Тяжесть ДТП в динамике
# Группируем ДТП и погибших по годам
yearly_severity = (
            sample_accidents
            .groupby(["year"])
            .agg(
                accident_number=("dead_count", "size"),  # Количество ДТП
                death_number=("dead_count", "sum")   # Сумма числа погибших
            )
            .reset_index()
)

   
# Рассчитываем число погибших на 1 ДТП
yearly_severity["death_per_accident"] = yearly_severity["death_number"] / yearly_severity["accident_number"]

# Номера месяцев и годы для оси X
plot_years = yearly_severity["year"].astype(str).tolist()

# Построение графика
locale.setlocale(locale.LC_ALL, 'Russian_Russia.1251')
mpl.rcParams['axes.formatter.use_locale'] = True
plt.figure(figsize=(16, 6))
plt.plot(range(len(plot_years)), yearly_severity["death_per_accident"], marker="o", linestyle="-", color="blue")

# Настройка оси X
plt.xticks(ticks=range(len(plot_years)), labels=plot_years, rotation=45, ha="right", fontsize=10)
plt.xlabel("Год", fontsize=12)
plt.ylabel("Погибших на 1 ДТП", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

"""Показатель достигает пика в 2017 году и последовательно снижается вплоть до минимума в 2024 году"""

#%% Тяжесть ДТП по регионам 
# Группируем ДТП и погибших по регионам
region_severity = (
    sample_accidents
    .groupby(["region"])
    .agg(
        accident_number=("dead_count", "size"),  # Количество ДТП
        death_number=("dead_count", "sum")   # Сумма числа погибших
    )
    .reset_index()
)

# Рассчитываем число погибших на 1 ДТП
region_severity["death_per_accident"] = region_severity["death_number"] / region_severity["accident_number"]

# Сортируем данные по убыванию "Погибших на ДТП"
region_severity = region_severity.sort_values(by="death_per_accident", ascending=False)

# Построение столбчатой диаграммы
plt.figure(figsize=(16, 6))
plt.bar(region_severity["region"], region_severity["death_per_accident"], color="blue", alpha=0.7)

# Настройка осей
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.xlabel("Регионы", fontsize=12)
plt.ylabel("Погибших на 1 ДТП", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)

plt.show()
"""Большое число регионов при небольшом районе данных дает ненадежные результаты: во многих регионах число погибших равно нулю или достигает 2-3 человек."""
#%% Карта с числом ДТП по регионам
"""Источник: https://github.com/hairymax/offline_russia_plotly/blob/main/data/russia_regions.geojson"""

region_geodata = gpd.read_file(r'../data/russia_regions.geojson')

region_accidents = (
    sample_accidents
    .groupby(["region"])
    .agg(
        accident_number=("id", "size"),  # Количество ДТП
    )
    .reset_index()
)
# Приводим названия регионов к общему формату 
# Словарь для специфических замен, которые нельзя автоматизировать

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

def normalize_region_name(name):
    """Автоматизация унификации названий регионов: замена 'область' на 'обл.'"""
    name = name.strip().lower()
    # Заменяем "область" на "обл."
    name = re.sub(r'\bобласть\b', 'обл.', name)
    # Заменяем "автономный округ" на "ао" для единообразия
    name = re.sub(r'\bавтономный округ\b', 'ао', name)
    return name

# Унификация
region_accidents['region'] = (region_accidents['region']
                             .apply(normalize_region_name)
                             .replace(region_mapping))
region_geodata['region'] = (region_geodata['region']
                           .apply(normalize_region_name))




# Объединяем геоданные с аварийностью по названию региона
region_geodata = region_geodata.merge(region_accidents, left_on="region", right_on="region", how="left")

# Заменяем NaN значения на 0
region_geodata["accident_number"].fillna(0, inplace=True)

# Преобразуем в проекцию, которая не искажает Россию
region_geodata = region_geodata.to_crs('EPSG:32646')

# Карта
fig, ax = plt.subplots(figsize=(12, 8))
region_geodata.plot(column="accident_number", cmap="Reds", linewidth=0.8, edgecolor="black",
            legend=True, ax=ax, legend_kwds={'shrink': 0.5})  # Уменьшаем шкалу (shrink=0.5)
region_geodata[region_geodata["accident_number"] == 0].plot(color='gray', ax=ax) # Отображение NaN как серого цвета
ax.axis("off")  # Убираем оси
plt.show()

#%% Карта с числом ДТП на 100 тысяч человек населения (социальный риск)

# Считаем ДТП на население
region_geodata["accident_rate"]=(region_geodata["accident_number"]/region_geodata["population"])*1000000

# Визуализация карты
fig, ax = plt.subplots(figsize=(12, 8))
region_geodata.plot(column="accident_rate", cmap="Reds", linewidth=0.8, edgecolor="black",
            legend=True, ax=ax, legend_kwds={'shrink': 0.5})  # Уменьшаем шкалу (shrink=0.5)

# Настройки карты
ax.axis("off")  # Убираем оси
plt.show()

#%% Тяжесть ДТП по типам ДТП
# Группируем ДТП и погибших по типам ДТП
grouped_data = (
    sample_accidents
    .groupby(["category"])
    .agg(
        accident_number=("id", "size"),  # Количество ДТП
        dead_count=("dead_count", "sum"),   # Сумма числа погибших
        paticipant_count=("participants_count", "sum")
        
    )
    .reset_index()
)
grouped_data.head()
# Добавляем колонку "Погибших на ДТП"
grouped_data["fatality_rate"] = 100*(grouped_data["dead_count"] / grouped_data["paticipant_count"])
grouped_data["fatality_rate"] = grouped_data["fatality_rate"].round(1)
# Создаем таблицу для числа ДТП
# Сортируем данные по убыванию числа ДИП
grouped_data = grouped_data.sort_values(by="accident_number", ascending=False).reset_index(drop=True)
print(grouped_data)

# Создаем таблицу для погибших на 100 участников
# Сортируем данные по убыванию "Погибших на 100 участников"
grouped_data = grouped_data.sort_values(by="fatality_rate", ascending=False).reset_index(drop=True)
print(grouped_data)
#%% ДТП по состоянию освещенности
#Число ДТП по состоянию освещения

# Группируем ДТП и погибших по регионам

light_data = (
    sample_accidents
    .groupby(["light"])
    .agg(
        accident_number=("dead_count", "size"),  # Количество ДТП
        dead_count=("dead_count", "sum")   # Сумма числа погибших
    )
    .reset_index()
)

# Добавляем колонку "Погибших на ДТП"
light_data["fatality_rate"] = light_data["dead_count"] / light_data["accident_number"]
# Сортируем данные по убыванию "количество_ДТП"
light_data = light_data.sort_values(by="accident_number", ascending=False)

# Построение столбчатой диаграммы по количеству ДТП
plt.figure(figsize=(16, 6))
plt.bar(light_data["light"], light_data["accident_number"], color="blue", alpha=0.7)

# Убираем вертикальные линии сетки
plt.grid(False)

# Настройка осей
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.ylabel("Число ДТП", fontsize=12)

plt.show()

# Сортируем данные по убыванию тяжести ДТП в зависимости от освезения
light_data = light_data.sort_values(by="fatality_rate", ascending=False)
# Построение столбчатой диаграммы по тяжести ДТП
plt.figure(figsize=(16, 6))
plt.bar(light_data["light"], light_data["fatality_rate"], color="blue", alpha=0.7)

# Убираем вертикальные линии сетки
plt.grid(False)

# Настройка осей
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.ylabel("Погибших на 1 ДТП", fontsize=12)

plt.show()

#%% Число и тяжесть ДТП по погодным условиям