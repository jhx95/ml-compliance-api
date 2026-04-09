import pandas as pd
import numpy as np

# Генерируем 5000 "новых" записей (с небольшим сдвигом, чтобы был дрейф)
np.random.seed(123)
n = 5000

# Создаём небольшой дрейф: resolution_days увеличился, complaint_status изменился
data = {
    'complaint_status': np.random.choice([2, 3, 4, 5], n, p=[0.1, 0.2, 0.4, 0.3]),
    'num_reassignments': np.random.poisson(2, n),
    'has_photo_evidence': np.random.choice([0, 1], n, p=[0.3, 0.7]),
    'is_monsoon_season': np.random.choice([0, 1], n, p=[0.6, 0.4]),
    'resolution_days': np.random.gamma(5, 2, n).astype(int) + 5,  # увеличили
    'has_gps_location': np.random.choice([0, 1], n, p=[0.2, 0.8]),
    'repeat_complainant': np.random.choice([0, 1], n, p=[0.7, 0.3]),
    'severity': np.random.choice([0, 1, 2, 3], n, p=[0.1, 0.3, 0.4, 0.2]),
    'ward_code': np.random.randint(1, 20, n),
    'complaint_channel': np.random.choice([0, 1, 2, 3], n, p=[0.3, 0.3, 0.2, 0.2]),
    'citizen_satisfied': np.random.choice([0, 1], n, p=[0.4, 0.6])  # цель
}

df = pd.DataFrame(data)

# Сохраняем
df.to_csv('data/new_data.csv', index=False)
print(f"✅ Создано {len(df)} новых записей в data/new_data.csv")
print(df.head())