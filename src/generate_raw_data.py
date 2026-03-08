from pathlib import Path

import numpy as np
import pandas as pd


def generate_raw_data(rows: int = 4000) -> None:
    models = [
        "Dunk",
        "Jordan",
        "Jordan High",
        "Low",
        "Green",
        "Premium",
        "Exclusive",
    ]

    df = pd.DataFrame(
        {
            "id": range(rows),
            "model": np.random.choice(models, rows),
            "price": np.random.randint(30, 120, rows),
            "stock": np.random.randint(0, 100, rows),
        }
    )

    for _ in range(50):
        idx = np.random.randint(0, rows)
        df.loc[idx, "model"] = "None"

    for _ in range(50):
        idx = np.random.randint(0, rows)
        df.loc[idx, "price"] = np.nan

    for _ in range(50):
        idx = np.random.randint(0, rows)
        df.loc[idx, "stock"] = np.nan

    root_path = Path(__file__).parent.parent
    data_path = root_path / "data" / "raw_data.csv"
    df.to_csv(data_path, index=False)
    print(f"Raw data file created: {data_path}")


if __name__ == "__main__":
    generate_raw_data()