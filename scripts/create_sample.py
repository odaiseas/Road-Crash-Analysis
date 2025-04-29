import pandas as pd
import requests
import sqlite3
from io import StringIO

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
vehicles_download_url = get_yandex_download_url(participants_url)

accidents = pd.read_csv(accidents_download_url, sep=';')
participants = pd.read_csv(participants_download_url, sep=';')
vehicles = pd.read_csv(vehicles_download_url, sep=';')

# Создание SQLite базы
conn = sqlite3.connect('C:/Users/Admin/Road-Crash-Analysis/data/crash_database.db')

# Выборка 100 случайных ДТП
sample_accidents = accidents.sample(n=100, random_state=42)

# Объединение: accidents + participants
sample_participants = participants[participants['id'].isin(sample_accidents['id'])]
merged_accidents_participants = pd.merge(sample_accidents, sample_participants, on='id', how='left')

# Объединение: добавляем vehicles
sample_vehicles = vehicles[vehicles['participant_id'].isin(sample_participants['participant_id'])]
final_sample = pd.merge(merged_accidents_participants, sample_vehicles, on='participant_id', how='left')

# Сохранение результата
output_path = 'C:/Users/Admin/Road-Crash-Analysis/data/sample/sample_crashes.csv'
final_sample.to_csv(output_path, index=False)

print(f"Sample created with {len(final_sample)} rows and saved to {output_path}")