import pandas as pd
import requests
import sqlite3
from io import StringIO
import random
import csv

# Публичные ссылки на Яндекс.Диск с исходными данными
accidents_url = "https://disk.yandex.ru/d/yPdgwafR_2xElg"
participants_url = "https://disk.yandex.ru/d/YeyKLfXuETaEUQ"
vehicles_url = "https://disk.yandex.ru/d/NJApFGWb85CWVQ"

# Функция для получения прямой ссылки на скачивание
def get_yandex_download_url(public_url):
    base_url = "https://cloud-api.yandex.net/v1/disk/public/resources/download"
    response = requests.get(base_url, params={"public_key": public_url})
    return response.json()["href"]

# Загрузка данных
accidents_download_url = get_yandex_download_url(accidents_url)
participants_download_url = get_yandex_download_url(participants_url)
vehicles_download_url = get_yandex_download_url(vehicles_url)

accidents = pd.read_csv(accidents_download_url, sep=';')
participants = pd.read_csv(participants_download_url, sep=';')
vehicles = pd.read_csv(vehicles_download_url, sep=';')

# Создание SQLite базы
conn = sqlite3.connect('C:/Users/Admin/Road-Crash-Analysis/data/crash_database.db')

# Загрузка данных в SQLite
accidents.to_sql('accidents', conn, if_exists='replace', index=False)
participants.to_sql('participants', conn, if_exists='replace', index=False)
vehicles.to_sql('vehicles', conn, if_exists='replace', index=False)

# Установка зерна для воспроизводимости
random.seed(42)  # Устанавливаем зерно
conn.create_function("rand", 0, lambda: random.random())  # Переопределяем RANDOM() через Python

print("Participants head before sampling:")
print(participants.head())
print("Vehicles head before sampling:")
print(vehicles.head())


# 1. Выборка 2000 случайных ДТП
query_accidents = """
SELECT * FROM accidents
ORDER BY RANDOM()
LIMIT 1000
"""
sample_accidents = pd.read_sql_query(query_accidents, conn)

# 2. Выборка участников, связанных с выбранными ДТП
sample_accidents_ids = tuple(sample_accidents['id'])
query_participants = f"""
SELECT * FROM participants
WHERE accident_id IN {sample_accidents_ids}
"""
sample_participants = pd.read_sql_query(query_participants, conn)

# 3. Выборка транспорта, связанного с выбранными ДТП (исправлено)
query_vehicles = f"""
SELECT * FROM vehicles
WHERE accident_id IN {sample_accidents_ids}
"""
sample_vehicles = pd.read_sql_query(query_vehicles, conn)

# Закрытие соединения
conn.close()

# Сохранение трёх файлов
sample_accidents.to_csv('C:/Users/Admin/Road-Crash-Analysis/data/sample/sample_accidents.csv', index=False)
sample_participants.to_csv('C:/Users/Admin/Road-Crash-Analysis/data/sample/sample_participants.csv', index=False)
sample_vehicles.to_csv('C:/Users/Admin/Road-Crash-Analysis/data/sample/sample_vehicles.csv', index=False)

print(f"Created sample_accidents with {len(sample_accidents)} rows")
print(f"Created sample_participants with {len(sample_participants)} rows")
print(f"Created sample_vehicles with {len(sample_vehicles)} rows")