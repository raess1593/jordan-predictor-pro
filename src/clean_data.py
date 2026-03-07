import pandas as pd
import numpy as np
from pathlib import Path

def clean_data():
    root_path = Path(__file__).parent.parent
    raw_path = root_path / 'data' / 'raw_data.csv'
    cleaned_path = root_path / 'data' / 'cleaned_data.csv'

    df = pd.read_csv(raw_path)

    print("Cleaning data...")

    models = df['model'].value_counts().keys().tolist()

    if 'None' in models:
        models.remove('None')
        df['model'] = df['model'].replace('None', np.random.choice(models))

    for c in ['price', 'stock']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        df.loc[df[c] < 1, c] = np.nan

    df = df.dropna()

    for c in ['price', 'stock']:
        df[c] = df[c].round()

    df.to_csv(cleaned_path, index=True)
    print(f"Data cleaned and saved in {cleaned_path}")

if __name__ == "__main__":
    clean_data()