![Challenge Image](https://github.com/bodiwael/Triple-A-Nasa-Space-Apps-2025/blob/main/challenge.png)
# Exoplanet Classification System

AI-powered exoplanet candidate classification with a **FastAPI** backend and a **vanilla HTML/JS** frontend. It supports **single** and **batch** predictions, **ensemble consensus**, downloadable **CSV templates**, and a sleek, animated UI.

> Datasets currently wired end-to-end: **KEPLER** and **K2**
> Models supported per dataset (by naming convention): `XGBoost`, `LightGBM`, `CatBoost`, `Genesis_CNN`, `CNN-LSTM`, `Simple_MLP`

---

## Table of Contents

* [Abstract](#abstract)
* [Demo & Overview](#demo--overview)
* [Repository Structure](#repository-structure)
* [Prerequisites](#prerequisites)
* [Quick Start](#quick-start)
* [Model & Scaler Layout](#model--scaler-layout)
* [Frontend Usage](#frontend-usage)
* [API Reference](#api-reference)
* [Batch Inference & Templates](#batch-inference--templates)
* [Testing](#testing)
* [Troubleshooting](#troubleshooting)
* [Customization & Deployment](#customization--deployment)
* [Roadmap](#roadmap)
* [Acknowledgments & License](#acknowledgments--license)

---
## Abstract

A World Away: AI-Powered Exoplanet Insights Platform

We present an intelligent, end-to-end platform that converts NASA’s complex exoplanet datasets into actionable insights for researchers, educators, and innovators. Built around open data from **Kepler**, **K2**, and extensible to **TESS**, the system unifies data ingestion, preprocessing, interactive exploration, and AI-assisted classification to accelerate discovery and learning.

**What it does.** The platform collects, cleans, and harmonizes mission data, then provides interactive dashboards, batch/stream inference, and explorable visual analytics. Users can inspect orbital and stellar features, run **predictive models** to classify candidates, and receive **ensemble consensus** with confidence scores. Real-time alerts and anomaly flags highlight unusual patterns or promising targets.

**How it works.** A FastAPI backend integrates NASA and external scientific APIs, standardizes features, applies quality checks, and serves inference through a REST interface. A lightweight web UI enables non-technical users to query, visualize, and interpret results. Embedded **ML models** (tree-based and deep learning) perform pattern recognition and anomaly detection; an ensemble layer improves robustness and interpretability.

**Benefits.**

* **Democratizes** NASA open science by making data accessible and comprehensible.
* **Accelerates research** with reproducible pipelines and hypothesis-driven analytics.
* **Catalyzes innovation** for education, startups, and citizen science.
* **Engages the public** through intuitive, interactive tools.

**Tools & stack.** Python (pandas, NumPy, scikit-learn, TensorFlow), FastAPI, and JavaScript visualizations (Plotly/D3.js). Prototyping in Jupyter; collaboration via GitHub; deployable to cloud (AWS/GCP) with optional GPU acceleration.

**Impact.** By bridging raw telemetry and practical understanding, this solution advances NASA’s mission of expanding knowledge for the benefit of humanity—scaling exoplanet discovery, fostering STEM engagement, and enabling data-driven exploration.

---

## Demo & Overview

https://github.com/user-attachments/assets/91c23c21-8135-4c92-bb1c-87dc562daf82

* **Frontend:** `index.html` (no build tools required).
* **Backend:** `main.py` (FastAPI).
* **Prediction flow:**

  1. Choose a mission (KEPLER / K2).
  2. Enter object parameters or **Load Sample Data**.
  3. Run **Classify with All Models** to get per-model outputs + **Ensemble Consensus**.
  4. Optionally upload a **CSV** for batch predictions and export JSON results.


---

## Repository Structure

```
.
├── index.html                  # Animated UI + mission selection, forms, and results
├── main.py                     # FastAPI backend with single, all-model, and batch endpoints
├── requirements.txt            # Python dependencies (CPU-friendly defaults)
├── test_api.py                 # Local test suite for the API
├── Models Comparison.ipynb     # Compares Multiple Models for Multiple Datasets provided by Nasa Space Apps Challenge
└── saved_models/
    ├── metadata.json           # Informational metadata about datasets/models (read-only)
    ├── scaler_KEPLER.joblib    # Required scaler for KEPLER
    ├── scaler_K2.joblib        # Required scaler for K2
    └── <Model>_<DATASET>.{joblib|keras}  # Trained models (see below)
```

---

## Prerequisites

* **Python** 3.10–3.11 recommended (ensure compatibility with TensorFlow 2.15.0).
* OS packages for LightGBM/CatBoost may be required depending on your platform.
* A set of trained models and scalers placed under `saved_models/` following the **naming convention** below.

> If you don’t have trained models yet, you can still run the API and use endpoints like `/api/info` and `/api/datasets/...`, but prediction endpoints will return “model not found” until you add them.

---

## Quick Start

1. **Create & activate a virtual environment**

   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/Mac:
   source .venv/bin/activate
   ```

2. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Place models and scalers**

   * Put your files under `saved_models/` (see [Model & Scaler Layout](#model--scaler-layout)).

4. **Run the backend**

   ```bash
   # Option A: Python entry
   python main.py

   # Option B: Uvicorn directly
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

5. **Open the frontend**

   * Simply open `index.html` in your browser **or** serve it via any static server.
   * By default, the UI targets `http://localhost:8000/api`. To change that, edit:

     ```js
     // index.html
     const API_BASE = 'http://localhost:8000/api';
     ```

---

## Model & Scaler Layout

The backend loads models **on demand** and caches them. It expects:

* **Scalers**

  ```
  saved_models/scaler_KEPLER.joblib
  saved_models/scaler_K2.joblib
  ```
* **Models** — one file per model/dataset pair:

  ```
  saved_models/XGBoost_KEPLER.joblib
  saved_models/LightGBM_KEPLER.joblib
  saved_models/CatBoost_KEPLER.joblib
  saved_models/Genesis_CNN_KEPLER.keras
  saved_models/CNN-LSTM_KEPLER.keras
  saved_models/Simple_MLP_KEPLER.keras

  saved_models/XGBoost_K2.joblib
  saved_models/LightGBM_K2.joblib
  saved_models/CatBoost_K2.joblib
  saved_models/Genesis_CNN_K2.keras
  saved_models/CNN-LSTM_K2.keras
  saved_models/Simple_MLP_K2.keras
  ```

> **Important:** The **filename prefix** must exactly match the `AVAILABLE_MODELS` entries in `main.py`. If you add/remove models, update `AVAILABLE_MODELS` accordingly.
---
## Results
<img width="4461" height="3564" alt="random_forest_evaluation" src="https://github.com/user-attachments/assets/5f90849d-3b69-44fe-86bb-eb4003944fe0" />


---

## Frontend Usage

1. Choose **KEPLER** or **K2**.
2. Enter parameters (or click **Load Sample Data**).
3. Click **Classify with All Models**.
4. Review per-model outputs and the **Ensemble Consensus**.
5. For batch inference:

   * Prepare a CSV with exact column names for the chosen dataset.
   * Use the **Batch Prediction from CSV** section to upload.
   * Download consolidated JSON via **Download Full Results (JSON)**.

> A downloadable CSV template is exposed by the backend (see [Batch Inference & Templates](#batch-inference--templates)).

---

## API Reference

Base URL: `http://localhost:8000/api`

### GET `/info`

Basic API metadata, supported datasets, features, and declared model names.

```bash
curl http://localhost:8000/api/info
```

### GET `/datasets/{dataset}/features`

Required feature list for a dataset (`KEPLER` or `K2`).

```bash
curl http://localhost:8000/api/datasets/KEPLER/features
```

### GET `/datasets/{dataset}/template`

CSV template with multiple **sample rows** (UTF-8, downloadable).

```bash
curl -OJ http://localhost:8000/api/datasets/K2/template
# -> K2_template.csv
```

### POST `/predict/{dataset}/{model_name}`

Predict with **one** model.

* **Body (JSON):** flat map `{feature: value, ...}` using **exact** feature names from `/datasets/{dataset}/features`.

```bash
curl -X POST http://localhost:8000/api/predict/KEPLER/XGBoost \
  -H "Content-Type: application/json" \
  -d '{
        "koi_period": 3.52, "koi_duration": 2.5, "koi_depth": 500,
        "koi_prad": 1.2, "koi_teq": 800, "koi_steff": 5700,
        "koi_slogg": 4.5, "koi_srad": 1.0, "koi_model_snr": 15,
        "koi_tce_plnt_num": 1, "koi_score": 0.9
      }'
```

### POST `/predict-all/{dataset}`

Predict with **all** declared models and compute an **ensemble consensus**.

```bash
curl -X POST http://localhost:8000/api/predict-all/K2 \
  -H "Content-Type: application/json" \
  -d '{
        "pl_orbper": 10.5, "pl_rade": 2.1, "pl_bmasse": 5.2, "pl_eqt": 650,
        "st_teff": 5500, "st_logg": 4.3, "st_rad": 1.1, "st_mass": 1.0,
        "sy_pnum": 1, "sy_snum": 1
      }'
```

### POST `/predict-batch/{dataset}`

Batch prediction via **CSV upload**.

* **Form-Data:** key `file` with your `.csv` file.

```bash
curl -X POST http://localhost:8000/api/predict-batch/KEPLER \
  -F "file=@KEPLER_input.csv"
```

### GET `/models/status`

Discover which models exist on disk and which are loaded in cache.

```bash
curl http://localhost:8000/api/models/status
```

---

## Batch Inference & Templates

* **Templates** provide correct **columns** and a few **sample rows**:

  * `GET /api/datasets/KEPLER/template` → `KEPLER_template.csv`
  * `GET /api/datasets/K2/template` → `K2_template.csv`

* **Column Names (must match exactly)**

  * **KEPLER (11 features):**
    `koi_period, koi_duration, koi_depth, koi_prad, koi_teq, koi_steff, koi_slogg, koi_srad, koi_model_snr, koi_tce_plnt_num, koi_score`
  * **K2 (10 features):**
    `pl_orbper, pl_rade, pl_bmasse, pl_eqt, st_teff, st_logg, st_rad, st_mass, sy_pnum, sy_snum`

* Missing values are imputed with the **column mean** at inference time.

---

## Testing

Run the local test suite after starting the server:

```bash
python main.py   # or: uvicorn main:app --reload
python test_api.py
```

**Notes**

* `test_api.py` includes:

  * `/info`
  * `/datasets/{dataset}/features`
  * `/predict/{dataset}/{model}`
  * `/predict-all/{dataset}`
  * `/models/status`

* If you see a failure such as:

  * `Single Model (Random_Forest/K2)` → **By design**, `AVAILABLE_MODELS` does **not** include `"Random_Forest"`.

    * Either **add** a matching model file `saved_models/Random_Forest_K2.joblib` and include the string `"Random_Forest"` in `AVAILABLE_MODELS['K2']`, **or** change the test to a declared model name (e.g., `"XGBoost"`).

---

## Troubleshooting

* **CORS / Frontend cannot reach API**

  * The backend enables permissive CORS. Ensure `API_BASE` in `index.html` points to the correct host/port.

* **“Scaler not found for {dataset}”**

  * Ensure `saved_models/scaler_{DATASET}.joblib` exists.

* **“Model {name} not found for {dataset}”**

  * Ensure a file named `{ModelName}_{DATASET}.joblib` **or** `{ModelName}_{DATASET}.keras` exists.
  * Ensure `AVAILABLE_MODELS[DATASET]` includes the exact model name.

* **TensorFlow / LightGBM install issues**

  * Use Python 3.10/3.11 and platform-compatible wheels.
  * For CPU-only environments, the provided versions are typically sufficient.

* **CSV upload errors**

  * Check the **exact** column headers (no extra spaces). The server trims whitespace from column names but field names must still match.

---

## Customization & Deployment

* **Changing models:** Edit `AVAILABLE_MODELS` and drop in new files in `saved_models/`.
* **Feature lists:** Edit `DATASET_FEATURES` for each dataset.
* **Frontend API target:** Update `API_BASE` in `index.html`.

### Deployment Ideas

* **Uvicorn/Gunicorn (Linux)**

  ```bash
  pip install "uvicorn[standard]" gunicorn
  gunicorn -k uvicorn.workers.UvicornWorker -w 2 -b 0.0.0.0:8000 main:app
  ```

* **Docker (example)**

  ```dockerfile
  FROM python:3.11-slim
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install --no-cache-dir -r requirements.txt
  COPY . .
  EXPOSE 8000
  CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
  ```

  Build & run:

  ```bash
  docker build -t exoplanet-api .
  docker run -p 8000:8000 -v $(pwd)/saved_models:/app/saved_models exoplanet-api
  ```

> Static hosting for the frontend is trivial (any static server). The backend is a standard ASGI app—deploy on your preferred platform (VM, Docker, Render, Railway, Fly.io, etc.).

---

## Roadmap

* Add **TESS** end-to-end (present in `metadata.json` but not yet wired in UI/API config).
* Add **/health** endpoint and richer **/models/status** diagnostics.
* Optional: JWT-protected endpoints and rate limiting.
* Optional: Swap in **streaming** progress updates for large batch jobs.

---

## Acknowledgments & License

* **Data & Inspiration:** NASA Kepler/K2 programs and the broader exoplanet research community.
* **Libraries:** FastAPI, scikit-learn, TensorFlow, XGBoost, LightGBM, CatBoost, pandas, numpy.

Licensed under [MIT](https://github.com/bodiwael/Triple-A-Nasa-Space-Apps-2025/blob/main/LICENSE)

---

## Citation

If you use this software in academic work, please cite this repository and the underlying missions/datasets as appropriate (Kepler/K2 references).
