# 👟 jordan-predictor-pro

Simple, reproducible *Machine Learning* pipeline to **validate**, **clean**, **train**, and **predict** sneaker prices from stock values.

---

## 🗂️ Project structure

- `data/` 📁: raw and cleaned CSV data (**tracked with DVC**)
- `src/` 🧠: Python scripts for generation, validation, cleaning, training, and prediction
- `notebooks/` 📓: optional exploration notebooks
- `tests/` 🧪: test placeholders
- `dvc.yaml` 🔁: pipeline definition

---

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate data and train model:**
   ```bash
   python src/generate_raw_data.py
   python src/clean_data.py
   python src/train.py
   ```
   
   Or run the full pipeline:
   ```bash
   dvc repro
   ```

3. **Use the API or CLI for predictions** 🔮 (see below)

---

## Pipeline (DVC)

Run full pipeline:

```bash
dvc repro
```

Run step-by-step:

```bash
python src/validate_data.py
python src/clean_data.py
python src/train.py
```

---

## 🧪 Generate synthetic raw data

```bash
python src/generate_raw_data.py
```

### CLI

```bash
python src/predict.py --stock 20
```

---

### API (FastAPI) 🌐

**⚠️ Important:** Train the model first (see Quick Start above)!

Start the API server:

```bash
uvicorn src.api:app --reload
```

![Example: ](assets/f1.png)
![](assets/f2.png)

Open your browser at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for interactive Swagger UI.

**Endpoints:**
- `GET /` - Welcome message
- `GET /health` - Check if model is loaded
- `GET /predict?stock=<value>` - Get price prediction

Example requests:

```bash
# Check health
curl "http://127.0.0.1:8000/health"

# Get prediction
curl "http://127.0.0.1:8000/predict?stock=20"
```
