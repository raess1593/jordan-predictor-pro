import pandas as pd
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

def train_model():

    #mlflow.create_experiment("my_model")
    mlflow.set_experiment("my_model")

    root_path = Path(__file__).parent.parent
    data_path = root_path / 'data' / 'cleaned_data.csv'
    df = pd.read_csv(data_path)

    X = df['stock'].values.reshape(-1, 1)
    y = df['price'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, shuffle=True, test_size=0.2
    )

    with mlflow.start_run(run_name="RamdomForest"):
        params = {
            'n_estimators': 100,
            'max_depth': 5,
            'min_samples_split': 2
        }
        rf = RandomForestRegressor(**params)
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)

        mlflow.log_params(params)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)

        mlflow.sklearn.log_model(rf, "price_by_stock_model")
        print(f"Modelo entrenado. MAE: {mae:.2f} -- RMSE: {rmse:.2f}")

if __name__ == "__main__":
    train_model()