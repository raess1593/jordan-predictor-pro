import pandas as pd
import numpy as np
from pathlib import Path

n=4000
models=['Dunk', 'Jordan', 'Jordan High', 'Low', 'Green', 'Premium', 'Exclussive']

df = pd.DataFrame({
    'id': range(n),
    'model': np.random.choice(models, n),
    'price': np.random.randint(30, 120, n),
    'stock': np.random.randint(0, 100, n)
})

for _ in range(50):
    x = np.random.randint(0, n)
    df.loc[x, 'model'] = 'None'
for _ in range(50):
    x = np.random.randint(0, n)
    df.loc[x, 'price'] = np.nan
for _ in range(50):
    x = np.random.randint(0, n)
    df.loc[x, 'stock'] = np.nan

root_path = Path(__file__).parent.parent
data_path = root_path / 'data' / 'raw_data.csv'
df.to_csv(data_path, index=False)
print("Raw data file correctly created")