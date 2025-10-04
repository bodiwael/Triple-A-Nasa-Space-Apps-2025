# 🪐 Exoplanet Classification Web Application

Advanced space-themed FastAPI application for exoplanet classification using multiple ML models.

## 📁 Project Structure

```
exoplanet-classifier/
├── main.py                 # FastAPI backend
├── index.html             # Space-themed frontend
├── saved_models/          # Your trained models (download from Kaggle)
│   ├── Random_Forest_KEPLER.joblib
│   ├── XGBoost_KEPLER.joblib
│   ├── CatBoost_KEPLER.joblib
│   ├── Genesis_CNN_KEPLER.keras
│   ├── scaler_KEPLER.joblib
│   ├── Random_Forest_K2.joblib
│   ├── XGBoost_K2.joblib
│   ├── ... (all your model files)
│   └── metadata.json
└── requirements.txt       # Python dependencies
```

## 🚀 Setup Instructions

### 1. Download Your Models from Kaggle

After running your training notebook in Kaggle, download all files:

```bash
# In your Kaggle notebook, run the download script
# Then download the saved_models.zip file
# Extract it in your project directory
```

### 2. Install Dependencies

```bash
# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install fastapi uvicorn joblib scikit-learn numpy pandas tensorflow xgboost lightgbm catboost python-multipart
```

Or create a `requirements.txt`:

```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
joblib==1.3.2
scikit-learn==1.3.2
numpy==1.24.3
pandas==2.1.3
tensorflow==2.15.0
xgboost==2.0.2
lightgbm==4.1.0
catboost==1.2.2
python-multipart==0.0.6
```

Then install:

```bash
pip install -r requirements.txt
```

### 3. Verify File Structure

Make sure you have:
- ✅ `main.py` (FastAPI backend)
- ✅ `index.html` (Frontend)
- ✅ `saved_models/` folder with all `.joblib`, `.keras`, and `metadata.json` files

### 4. Run the Application

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Open in Browser

🌐 **Web Interface**: http://localhost:8000  
📖 **API Documentation**: http://localhost:8000/docs  
📊 **Alternative API Docs**: http://localhost:8000/redoc

## 🎮 How to Use

### Option 1: Single Prediction (Web Interface)

1. **Select a Mission**: Choose KEPLER or K2
2. **Enter Parameters**: Fill in all feature values (or click "Load Sample Data")
3. **Classify**: Click "🚀 Classify with All Models"
4. **View Results**: See predictions from all 8 models + ensemble consensus

### Option 2: Batch Prediction (CSV Upload)

1. Select a mission
2. Prepare a CSV file with the required features
3. Click the file upload area
4. Results will be downloaded as JSON

### Option 3: API Usage (Programmatic)

#### Get All Predictions for an Object

```python
import requests

url = "http://localhost:8000/api/predict-all/KEPLER"

data = {
    "koi_period": 3.52,
    "koi_duration": 2.5,
    "koi_depth": 500,
    "koi_prad": 1.2,
    "koi_teq": 800,
    "koi_steff": 5700,
    "koi_slogg": 4.5,
    "koi_srad": 1.0,
    "koi_model_snr": 15,
    "koi_tce_plnt_num": 1,
    "koi_score": 0.9
}

response = requests.post(url, json=data)
result = response.json()

print(f"Consensus: {result['consensus']['prediction_label']}")
print(f"Confidence: {result['consensus']['confidence'] * 100:.1f}%")

for model in result['results']:
    print(f"{model['model']}: {model['prediction_label']} ({model['confidence']*100:.1f}%)")
```

#### Single Model Prediction

```python
url = "http://localhost:8000/api/predict/KEPLER/XGBoost"

response = requests.post(url, json=data)
result = response.json()

print(f"Prediction: {result['prediction_label']}")
print(f"Confidence: {result['confidence'] * 100:.1f}%")
```

#### Check Model Status

```python
response = requests.get("http://localhost:8000/api/models/status")
print(response.json())
```

## 📊 Required Features for Each Dataset

### KEPLER (11 features)
- `koi_period` - Orbital Period (days)
- `koi_duration` - Transit Duration (hours)
- `koi_depth` - Transit Depth (ppm)
- `koi_prad` - Planetary Radius (Earth radii)
- `koi_teq` - Equilibrium Temperature (K)
- `koi_steff` - Stellar Effective Temperature (K)
- `koi_slogg` - Stellar Surface Gravity
- `koi_srad` - Stellar Radius (Solar radii)
- `koi_model_snr` - Transit Signal-to-Noise Ratio
- `koi_tce_plnt_num` - Planet Number
- `koi_score` - Disposition Score

### K2 (10 features)
- `pl_orbper` - Orbital Period (days)
- `pl_rade` - Planet Radius (Earth radii)
- `pl_bmasse` - Planet Mass (Earth masses)
- `pl_eqt` - Equilibrium Temperature (K)
- `st_teff` - Stellar Effective Temperature (K)
- `st_logg` - Stellar Surface Gravity
- `st_rad` - Stellar Radius (Solar radii)
- `st_mass` - Stellar Mass (Solar masses)
- `sy_pnum` - Number of Planets in System
- `sy_snum` - Number of Stars in System

## 🎨 Features

✨ **Space-Themed UI** with animated stars and cosmic effects  
🤖 **8 ML Models** per dataset (Random Forest, XGBoost, CatBoost, LightGBM, Gradient Boosting, Genesis CNN, CNN-LSTM, Simple MLP)  
🎯 **Ensemble Consensus** - Majority voting across all models  
📊 **Visual Results** - Beautiful cards showing each model's prediction  
📁 **Batch Processing** - Upload CSV for multiple predictions  
⚡ **Fast API** - Sub-second response times  
🔄 **Model Caching** - Models loaded once and reused  

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/docs` | GET | Interactive API documentation |
| `/api/info` | GET | API information and available models |
| `/api/datasets/{dataset}/features` | GET | Get required features for a dataset |
| `/api/predict/{dataset}/{model}` | POST | Predict with a specific model |
| `/api/predict-all/{dataset}` | POST | Predict with all models + consensus |
| `/api/predict-batch/{dataset}` | POST | Batch predictions from CSV |
| `/api/models/status` | GET | Check which models are loaded |

## 🐛 Troubleshooting

### Models not found
- Ensure `saved_models/` folder exists
- Check that all `.joblib` and `.keras` files are present
- Verify file names match: `ModelName_DATASET.joblib` or `ModelName_DATASET.keras`

### Port already in use
```bash
# Use a different port
uvicorn main:app --port 8001
```

### CORS errors
- The backend already has CORS enabled
- Make sure you're accessing from `localhost` or update `allow_origins` in `main.py`

### Module not found
```bash
# Reinstall all dependencies
pip install -r requirements.txt
```

## 🎯 Example CSV Format

**kepler_data.csv**:
```csv
koi_period,koi_duration,koi_depth,koi_prad,koi_teq,koi_steff,koi_slogg,koi_srad,koi_model_snr,koi_tce_plnt_num,koi_score
3.52,2.5,500,1.2,800,5700,4.5,1.0,15,1,0.9
5.2,3.1,450,1.5,750,5600,4.4,1.1,18,1,0.85
```

**k2_data.csv**:
```csv
pl_orbper,pl_rade,pl_bmasse,pl_eqt,st_teff,st_logg,st_rad,st_mass,sy_pnum,sy_snum
10.5,2.1,5.2,650,5500,4.3,1.1,1.0,1,1
8.3,1.8,4.5,700,5600,4.4,1.0,0.95,2,1
```

## 📈 Performance

- **Response Time**: ~50-200ms per prediction (all 8 models)
- **Batch Processing**: ~500ms for 10 samples
- **Model Loading**: First request is slower (~2-3s), then cached
- **Memory Usage**: ~2-4GB with all models loaded

## 🚀 Deployment Options

### Local Development
```bash
python main.py
```

### Production (Docker)
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Cloud Platforms
- **Heroku**: Deploy with Procfile
- **AWS**: Use Elastic Beanstalk or EC2
- **Google Cloud**: Deploy to Cloud Run
- **Azure**: Use Azure App Service

## 📝 License

MIT License - Feel free to use and modify!

## 🌟 Credits

Built with ❤️ using FastAPI, TensorFlow, scikit-learn, and XGBoost.

---

**Happy Planet Hunting! 🪐✨**