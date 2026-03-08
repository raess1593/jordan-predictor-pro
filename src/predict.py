import argparse

import mlflow
import mlflow.sklearn
import numpy as np


def get_latest_model(experiment_name):
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment {experiment_name} was not found")
    
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="status = 'FINISHED'",
        order_by=["start_time DESC"],
        max_results=1
    )

    if runs.empty:
        raise ValueError("There are no runs yet")
    
    last_run_id = runs.iloc[0]['run_id']
    model_uri = f"runs:/{last_run_id}/price_by_stock_model"

    model = mlflow.sklearn.load_model(model_uri)
    return model

def run_inference(stock_value):
    model = get_latest_model("my_model")
    try:
        input_data = np.array([[stock_value]], dtype=float)
        pred = model.predict(input_data)

        print(f"Estimated price for stock {stock_value}: {pred[0]:.2f}")

    except Exception as e:
        print(f"Error during inference: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description="Run model inference")
    parser.add_argument("--stock", type=float, default=20, help="Stock value")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_inference(args.stock)