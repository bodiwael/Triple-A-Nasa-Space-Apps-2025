# main.py - FastAPI Backend for Exoplanet Classification
# Place this file in the same directory as your saved_models folder

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Optional
import joblib
import numpy as np
import pandas as pd
from tensorflow import keras
from pathlib import Path
import json
import io

app = FastAPI(title="Exoplanet Classification API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# CONFIGURATION
# ============================================================================

MODELS_DIR = Path('saved_models')

# Dataset configurations
DATASET_FEATURES = {
    'KEPLER': [
        'koi_period', 'koi_duration', 'koi_depth', 'koi_prad', 'koi_teq',
        'koi_steff', 'koi_slogg', 'koi_srad', 'koi_model_snr', 
        'koi_tce_plnt_num', 'koi_score'
    ],
    'K2': [
        'pl_orbper', 'pl_rade', 'pl_bmasse', 'pl_eqt',
        'st_teff', 'st_logg', 'st_rad', 'st_mass',
        'sy_pnum', 'sy_snum'
    ]
}

# Sample data for CSV templates
SAMPLE_CSV_DATA = {
    'KEPLER': [
        {'koi_period': 3.52, 'koi_duration': 2.5, 'koi_depth': 500, 'koi_prad': 1.2, 
         'koi_teq': 800, 'koi_steff': 5700, 'koi_slogg': 4.5, 'koi_srad': 1.0, 
         'koi_model_snr': 15, 'koi_tce_plnt_num': 1, 'koi_score': 0.9},
        {'koi_period': 10.8, 'koi_duration': 3.2, 'koi_depth': 800, 'koi_prad': 2.1, 
         'koi_teq': 650, 'koi_steff': 5500, 'koi_slogg': 4.3, 'koi_srad': 1.1, 
         'koi_model_snr': 20, 'koi_tce_plnt_num': 1, 'koi_score': 0.85},
        {'koi_period': 1.2, 'koi_duration': 1.5, 'koi_depth': 200, 'koi_prad': 0.8, 
         'koi_teq': 1200, 'koi_steff': 6000, 'koi_slogg': 4.6, 'koi_srad': 0.9, 
         'koi_model_snr': 12, 'koi_tce_plnt_num': 2, 'koi_score': 0.75}
    ],
    'K2': [
        {'pl_orbper': 10.5, 'pl_rade': 2.1, 'pl_bmasse': 5.2, 'pl_eqt': 650, 
         'st_teff': 5500, 'st_logg': 4.3, 'st_rad': 1.1, 'st_mass': 1.0, 
         'sy_pnum': 1, 'sy_snum': 1},
        {'pl_orbper': 25.3, 'pl_rade': 3.5, 'pl_bmasse': 12.5, 'pl_eqt': 450, 
         'st_teff': 5200, 'st_logg': 4.2, 'st_rad': 1.3, 'st_mass': 1.1, 
         'sy_pnum': 2, 'sy_snum': 1},
        {'pl_orbper': 5.2, 'pl_rade': 1.5, 'pl_bmasse': 3.2, 'pl_eqt': 900, 
         'st_teff': 5800, 'st_logg': 4.4, 'st_rad': 1.0, 'st_mass': 0.95, 
         'sy_pnum': 1, 'sy_snum': 1}
    ]
}

# Available models for each dataset
AVAILABLE_MODELS = {
    'KEPLER': [
        'XGBoost', 'LightGBM', 'CatBoost', 
        'Genesis_CNN', 'CNN-LSTM', 'Simple_MLP'
    ],
    'K2': [
        'XGBoost', 'LightGBM', 'CatBoost', 
        'Genesis_CNN', 'CNN-LSTM', 'Simple_MLP'
    ]
}

# ============================================================================
# GLOBAL MODEL CACHE
# ============================================================================

models_cache = {}
scalers_cache = {}

def load_model(model_name: str, dataset: str):
    """Load a model and its scaler from disk (with caching)"""
    cache_key = f"{model_name}_{dataset}"
    
    # Check cache
    if cache_key in models_cache:
        return models_cache[cache_key], scalers_cache.get(f"scaler_{dataset}")
    
    # Load scaler
    scaler_key = f"scaler_{dataset}"
    if scaler_key not in scalers_cache:
        scaler_path = MODELS_DIR / f"scaler_{dataset}.joblib"
        if not scaler_path.exists():
            raise HTTPException(status_code=404, detail=f"Scaler not found for {dataset}")
        scalers_cache[scaler_key] = joblib.load(scaler_path)
    
    # Load model
    # Try joblib first (traditional ML)
    joblib_path = MODELS_DIR / f"{model_name}_{dataset}.joblib"
    if joblib_path.exists():
        model = joblib.load(joblib_path)
        models_cache[cache_key] = model
        return model, scalers_cache[scaler_key]
    
    # Try keras (deep learning)
    keras_path = MODELS_DIR / f"{model_name}_{dataset}.keras"
    if keras_path.exists():
        model = keras.models.load_model(keras_path)
        models_cache[cache_key] = model
        return model, scalers_cache[scaler_key]
    
    raise HTTPException(status_code=404, detail=f"Model {model_name} not found for {dataset}")

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class PredictionInput(BaseModel):
    dataset: str
    features: Dict[str, float]
    
class BatchPredictionResponse(BaseModel):
    dataset: str
    total_samples: int
    results: List[Dict]

# ============================================================================
# ROUTES
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML page"""
    html_file = Path("index.html")
    if html_file.exists():
        return html_file.read_text(encoding='utf-8')
    return """
    <html>
        <head><title>Exoplanet Classification</title></head>
        <body>
            <h1>Exoplanet Classification API</h1>
            <p>API is running! Visit <a href="/docs">/docs</a> for API documentation.</p>
        </body>
    </html>
    """

@app.get("/api/info")
async def get_info():
    """Get API information and available models"""
    return {
        "api_name": "Exoplanet Classification API",
        "version": "1.0.0",
        "datasets": list(DATASET_FEATURES.keys()),
        "available_models": AVAILABLE_MODELS,
        "features": DATASET_FEATURES
    }

@app.get("/api/datasets/{dataset}/features")
async def get_dataset_features(dataset: str):
    """Get required features for a specific dataset"""
    dataset = dataset.upper()
    if dataset not in DATASET_FEATURES:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset} not found")
    
    return {
        "dataset": dataset,
        "features": DATASET_FEATURES[dataset],
        "feature_count": len(DATASET_FEATURES[dataset])
    }

@app.get("/api/datasets/{dataset}/template")
async def download_csv_template(dataset: str):
    """Download a CSV template with sample data for the specified dataset"""
    dataset = dataset.upper()
    
    if dataset not in DATASET_FEATURES:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset} not found")
    
    # Create DataFrame with sample data
    df = pd.DataFrame(SAMPLE_CSV_DATA[dataset])
    
    # Convert to CSV string
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_string = csv_buffer.getvalue()
    
    # Return as downloadable file with proper encoding
    return StreamingResponse(
        io.BytesIO(csv_string.encode('utf-8')),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename={dataset}_template.csv",
            "Access-Control-Expose-Headers": "Content-Disposition"
        }
    )

@app.post("/api/predict/{dataset}/{model_name}")
async def predict_single(dataset: str, model_name: str, input_data: Dict[str, float]):
    """Make a single prediction using a specific model"""
    dataset = dataset.upper()
    
    # Validate dataset
    if dataset not in DATASET_FEATURES:
        raise HTTPException(status_code=400, detail=f"Invalid dataset: {dataset}")
    
    # Validate model
    if model_name not in AVAILABLE_MODELS.get(dataset, []):
        raise HTTPException(status_code=400, detail=f"Model {model_name} not available for {dataset}")
    
    # Get required features
    required_features = DATASET_FEATURES[dataset]
    
    # Validate input features
    missing_features = set(required_features) - set(input_data.keys())
    if missing_features:
        raise HTTPException(
            status_code=400, 
            detail=f"Missing features: {list(missing_features)}"
        )
    
    try:
        # Load model and scaler
        model, scaler = load_model(model_name, dataset)
        
        # Prepare input as DataFrame to preserve feature names
        X = pd.DataFrame([[input_data[f] for f in required_features]], columns=required_features)
        
        # Scale
        X_scaled = scaler.transform(X)
        
        # Predict
        is_keras = keras_path = MODELS_DIR / f"{model_name}_{dataset}.keras"
        is_keras = is_keras.exists()
        
        if is_keras:
            # Deep learning model
            probabilities = model.predict(X_scaled, verbose=0)[0]
            prediction = int(np.argmax(probabilities))
            prob_planet = float(probabilities[1])
            prob_not_planet = float(probabilities[0])
        else:
            # Traditional ML model
            prediction = int(model.predict(X_scaled)[0])
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_scaled)[0]
                prob_not_planet = float(probabilities[0])
                prob_planet = float(probabilities[1])
            else:
                prob_planet = None
                prob_not_planet = None
        
        return {
            "model": model_name,
            "dataset": dataset,
            "prediction": prediction,
            "prediction_label": "CONFIRMED PLANET" if prediction == 1 else "NOT A PLANET",
            "confidence": prob_planet if prob_planet else None,
            "probabilities": {
                "planet": prob_planet,
                "not_planet": prob_not_planet
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/api/predict-all/{dataset}")
async def predict_all_models(dataset: str, input_data: Dict[str, float]):
    """Make predictions using ALL models for a specific dataset"""
    dataset = dataset.upper()
    
    # Validate dataset
    if dataset not in DATASET_FEATURES:
        raise HTTPException(status_code=400, detail=f"Invalid dataset: {dataset}")
    
    # Get required features
    required_features = DATASET_FEATURES[dataset]
    
    # Validate input features
    missing_features = set(required_features) - set(input_data.keys())
    if missing_features:
        raise HTTPException(
            status_code=400, 
            detail=f"Missing features: {list(missing_features)}"
        )
    
    results = []
    
    # Prepare input once as DataFrame to preserve feature names
    X = pd.DataFrame([[input_data[f] for f in required_features]], columns=required_features)
    
    # Get scaler
    scaler_key = f"scaler_{dataset}"
    if scaler_key not in scalers_cache:
        scaler_path = MODELS_DIR / f"scaler_{dataset}.joblib"
        scalers_cache[scaler_key] = joblib.load(scaler_path)
    
    scaler = scalers_cache[scaler_key]
    X_scaled = scaler.transform(X)
    
    # Predict with all models
    for model_name in AVAILABLE_MODELS[dataset]:
        try:
            model, _ = load_model(model_name, dataset)
            
            # Check if it's a keras model
            is_keras = (MODELS_DIR / f"{model_name}_{dataset}.keras").exists()
            
            if is_keras:
                probabilities = model.predict(X_scaled, verbose=0)[0]
                prediction = int(np.argmax(probabilities))
                prob_planet = float(probabilities[1])
                prob_not_planet = float(probabilities[0])
            else:
                # Convert to numpy array for traditional ML models to avoid sklearn warnings
                X_array = X_scaled if isinstance(X_scaled, np.ndarray) else X_scaled.values
                
                prediction = int(model.predict(X_array)[0])
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(X_array)[0]
                    prob_not_planet = float(probabilities[0])
                    prob_planet = float(probabilities[1])
                else:
                    prob_planet = 0.5
                    prob_not_planet = 0.5
            
            results.append({
                "model": model_name,
                "prediction": prediction,
                "prediction_label": "CONFIRMED PLANET" if prediction == 1 else "NOT A PLANET",
                "confidence": prob_planet,
                "probabilities": {
                    "planet": prob_planet,
                    "not_planet": prob_not_planet
                }
            })
            
        except Exception as e:
            print(f"Error predicting with {model_name}: {str(e)}")
            results.append({
                "model": model_name,
                "error": str(e),
                "prediction": -1,
                "prediction_label": "ERROR",
                "confidence": 0.0,
                "probabilities": {
                    "planet": 0.0,
                    "not_planet": 0.0
                }
            })
    
    # Calculate consensus
    valid_predictions = [r for r in results if r.get('prediction') is not None and r.get('prediction') != -1]
    if valid_predictions:
        planet_votes = sum(1 for r in valid_predictions if r['prediction'] == 1)
        not_planet_votes = sum(1 for r in valid_predictions if r['prediction'] == 0)
        total_votes = len(valid_predictions)
        consensus_prediction = 1 if planet_votes > not_planet_votes else 0
        consensus_confidence = planet_votes / total_votes if consensus_prediction == 1 else not_planet_votes / total_votes
    else:
        consensus_prediction = None
        consensus_confidence = None
        planet_votes = 0
        not_planet_votes = 0
        total_votes = 0
    
    return {
        "dataset": dataset,
        "input_features": input_data,
        "model_count": len(valid_predictions),
        "results": sorted(results, key=lambda x: x.get('confidence', 0), reverse=True),
        "consensus": {
            "prediction": consensus_prediction,
            "prediction_label": "CONFIRMED PLANET" if consensus_prediction == 1 else "NOT A PLANET" if consensus_prediction == 0 else "UNCERTAIN",
            "confidence": consensus_confidence,
            "votes_planet": planet_votes,
            "votes_not_planet": not_planet_votes,
            "total_models": total_votes
        }
    }

@app.post("/api/predict-batch/{dataset}")
async def predict_batch(dataset: str, file: UploadFile = File(...)):
    """Upload a CSV file and get predictions from all models"""
    dataset = dataset.upper()
    
    if dataset not in DATASET_FEATURES:
        raise HTTPException(status_code=400, detail=f"Invalid dataset: {dataset}")
    
    # Read CSV
    try:
        contents = await file.read()
        # Handle different encodings
        try:
            csv_string = contents.decode('utf-8')
        except UnicodeDecodeError:
            csv_string = contents.decode('latin-1')
        
        df = pd.read_csv(io.StringIO(csv_string))
        
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV: {str(e)}")
    
    # Validate features
    required_features = DATASET_FEATURES[dataset]
    missing_features = set(required_features) - set(df.columns)
    if missing_features:
        raise HTTPException(
            status_code=400,
            detail=f"Missing columns in CSV: {list(missing_features)}. Found columns: {list(df.columns)}"
        )
    
    # Get scaler
    scaler_key = f"scaler_{dataset}"
    if scaler_key not in scalers_cache:
        scaler_path = MODELS_DIR / f"scaler_{dataset}.joblib"
        if not scaler_path.exists():
            raise HTTPException(status_code=404, detail=f"Scaler not found for {dataset}")
        scalers_cache[scaler_key] = joblib.load(scaler_path)
    
    scaler = scalers_cache[scaler_key]
    
    # Prepare data
    try:
        X = df[required_features].copy()
        # Handle missing values
        X = X.fillna(X.mean())
        X_scaled = scaler.transform(X)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing data: {str(e)}")
    
    batch_results = []
    
    # Process each row
    for idx, row_scaled in enumerate(X_scaled):
        row_input = row_scaled.reshape(1, -1)
        row_results = []
        
        # Predict with all models
        for model_name in AVAILABLE_MODELS[dataset]:
            try:
                model, _ = load_model(model_name, dataset)
                
                is_keras = (MODELS_DIR / f"{model_name}_{dataset}.keras").exists()
                
                if is_keras:
                    probabilities = model.predict(row_input, verbose=0)[0]
                    prediction = int(np.argmax(probabilities))
                    prob_planet = float(probabilities[1])
                else:
                    prediction = int(model.predict(row_input)[0])
                    if hasattr(model, 'predict_proba'):
                        probabilities = model.predict_proba(row_input)[0]
                        prob_planet = float(probabilities[1])
                    else:
                        prob_planet = 0.5
                
                row_results.append({
                    "model": model_name,
                    "prediction": prediction,
                    "confidence": prob_planet
                })
                
            except Exception as e:
                row_results.append({
                    "model": model_name,
                    "error": str(e)
                })
        
        # Calculate consensus for this row
        valid_predictions = [r['prediction'] for r in row_results if r.get('prediction') is not None]
        planet_votes = sum(valid_predictions)
        total_votes = len(valid_predictions)
        
        batch_results.append({
            "row_index": idx,
            "original_data": df.iloc[idx].to_dict(),
            "predictions": row_results,
            "consensus": {
                "prediction": 1 if planet_votes > total_votes / 2 else 0,
                "confidence": planet_votes / total_votes if total_votes > 0 else 0,
                "votes_planet": planet_votes,
                "votes_not_planet": total_votes - planet_votes
            }
        })
    
    return {
        "dataset": dataset,
        "total_samples": len(df),
        "results": batch_results
    }


@app.get("/api/models/status")
async def models_status():
    """Check which models are available and loaded"""
    status = {}
    
    for dataset in DATASET_FEATURES.keys():
        status[dataset] = {}
        for model_name in AVAILABLE_MODELS.get(dataset, []):
            joblib_path = MODELS_DIR / f"{model_name}_{dataset}.joblib"
            keras_path = MODELS_DIR / f"{model_name}_{dataset}.keras"
            
            exists = joblib_path.exists() or keras_path.exists()
            loaded = f"{model_name}_{dataset}" in models_cache
            
            status[dataset][model_name] = {
                "exists": exists,
                "loaded": loaded,
                "type": "keras" if keras_path.exists() else "joblib" if joblib_path.exists() else "unknown"
            }
    
    return {
        "models_status": status,
        "cache_size": len(models_cache),
        "scalers_loaded": len(scalers_cache)
    }

# ============================================================================
# STARTUP
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Pre-load metadata on startup"""
    print("🚀 Starting Exoplanet Classification API...")
    print(f"📂 Models directory: {MODELS_DIR.absolute()}")
    
    # Check if models directory exists
    if not MODELS_DIR.exists():
        print(f"⚠️  WARNING: Models directory not found!")
        print(f"   Please ensure '{MODELS_DIR}' exists with your trained models.")
    else:
        model_files = list(MODELS_DIR.glob('*'))
        print(f"✅ Found {len(model_files)} model files")
    
    print("✅ API Ready!")
    print("📖 API Docs: http://localhost:8000/docs")
    print("🌐 Web Interface: http://localhost:8000")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)