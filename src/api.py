from fastapi import FastAPI, HTTPException
import numpy as np

try:
    from .predict import get_latest_model
except ImportError:
    from predict import get_latest_model

app = FastAPI(
    title="Shoe Price Predictor API",
    description="API to predict shoe prices based on stock quantity",
    version="1.0.0"
)


@app.get("/")
def read_root():
    return {
        "message": "Welcome to Shoe Price Predictor API",
        "docs": "/docs",
        "predict_endpoint": "/predict?stock=<value>"
    }


@app.get("/health")
def health_check():
    """Check if the API and model are ready."""
    try:
        model = get_latest_model("my_model")
        return {
            "status": "healthy",
            "model_loaded": True
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "message": "No model found. Please train a model first: python src/train.py",
            "error": str(e)
        }


@app.get("/predict")
def predict(stock: float):
    """Predict shoe price based on stock quantity."""
    try:
        model = get_latest_model("my_model")
        input_data = np.array([[stock]], dtype=float)
        prediction = model.predict(input_data)

        return {
            "stock_input": stock,
            "estimated_price": round(float(prediction[0]), 2),
            "status": "success"
        }
    except ValueError as e:
        if "was not found" in str(e) or "no runs yet" in str(e).lower():
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Model not available",
                    "message": "No trained model found. Please train a model first.",
                    "instructions": [
                        "1. Run: python src/train.py",
                        "2. Or run full pipeline: dvc repro"
                    ]
                }
            )
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e)})