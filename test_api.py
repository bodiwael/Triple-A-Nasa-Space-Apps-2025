#!/usr/bin/env python3
"""
Test script for Exoplanet Classification API
Run this after starting the FastAPI server
"""

import requests
import json
from typing import Dict

# API Configuration
API_BASE = "http://localhost:8000/api"

# Sample data for testing
KEPLER_SAMPLE = {
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

K2_SAMPLE = {
    "pl_orbper": 10.5,
    "pl_rade": 2.1,
    "pl_bmasse": 5.2,
    "pl_eqt": 650,
    "st_teff": 5500,
    "st_logg": 4.3,
    "st_rad": 1.1,
    "st_mass": 1.0,
    "sy_pnum": 1,
    "sy_snum": 1
}

def print_header(text: str):
    """Print a formatted header"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80)

def test_api_info():
    """Test 1: Get API information"""
    print_header("TEST 1: API Information")
    
    response = requests.get(f"{API_BASE}/info")
    
    if response.status_code == 200:
        data = response.json()
        print("✅ API is running!")
        print(f"\nAPI Name: {data['api_name']}")
        print(f"Version: {data['version']}")
        print(f"Available Datasets: {', '.join(data['datasets'])}")
        print(f"\nAvailable Models:")
        for dataset, models in data['available_models'].items():
            print(f"  {dataset}: {len(models)} models")
            for model in models:
                print(f"    - {model}")
        return True
    else:
        print(f"❌ Failed to connect to API: {response.status_code}")
        return False

def test_dataset_features(dataset: str):
    """Test 2: Get dataset features"""
    print_header(f"TEST 2: {dataset} Dataset Features")
    
    response = requests.get(f"{API_BASE}/datasets/{dataset}/features")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ {dataset} requires {data['feature_count']} features:")
        for i, feature in enumerate(data['features'], 1):
            print(f"  {i}. {feature}")
        return True
    else:
        print(f"❌ Failed: {response.status_code}")
        return False

def test_single_model_prediction(dataset: str, model: str, input_data: Dict):
    """Test 3: Single model prediction"""
    print_header(f"TEST 3: Single Model Prediction ({model} on {dataset})")
    
    response = requests.post(
        f"{API_BASE}/predict/{dataset}/{model}",
        json=input_data
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Prediction successful!")
        print(f"\nModel: {data['model']}")
        print(f"Dataset: {data['dataset']}")
        print(f"Prediction: {data['prediction_label']}")
        if data.get('confidence'):
            print(f"Confidence: {data['confidence']*100:.1f}%")
        if data.get('probabilities'):
            print(f"\nProbabilities:")
            print(f"  Planet: {data['probabilities']['planet']*100:.1f}%")
            print(f"  Not Planet: {data['probabilities']['not_planet']*100:.1f}%")
        return True
    else:
        print(f"❌ Failed: {response.status_code}")
        print(f"Error: {response.text}")
        return False

def test_all_models_prediction(dataset: str, input_data: Dict):
    """Test 4: All models prediction with ensemble"""
    print_header(f"TEST 4: All Models Prediction on {dataset}")
    
    response = requests.post(
        f"{API_BASE}/predict-all/{dataset}",
        json=input_data
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Prediction successful with {data['model_count']} models!")
        
        # Consensus
        print(f"\n{'='*60}")
        print(f"🎯 ENSEMBLE CONSENSUS")
        print(f"{'='*60}")
        consensus = data['consensus']
        print(f"Prediction: {consensus['prediction_label']}")
        print(f"Confidence: {consensus['confidence']*100:.1f}%")
        print(f"Votes - Planet: {consensus['votes_planet']} | Not Planet: {consensus['votes_not_planet']}")
        
        # Individual models
        print(f"\n{'='*60}")
        print(f"🤖 INDIVIDUAL MODEL PREDICTIONS")
        print(f"{'='*60}")
        
        # Sort by confidence
        sorted_results = sorted(
            data['results'], 
            key=lambda x: x.get('confidence', 0), 
            reverse=True
        )
        
        for result in sorted_results:
            if result.get('error'):
                print(f"❌ {result['model']}: Error - {result['error']}")
            else:
                emoji = "🟢" if result['prediction'] == 1 else "🔴"
                print(f"{emoji} {result['model']:20s} | {result['prediction_label']:20s} | Confidence: {result['confidence']*100:5.1f}%")
        
        return True
    else:
        print(f"❌ Failed: {response.status_code}")
        print(f"Error: {response.text}")
        return False

def test_models_status():
    """Test 5: Check models status"""
    print_header("TEST 5: Models Status")
    
    response = requests.get(f"{API_BASE}/models/status")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Status check successful!")
        print(f"\nCache Size: {data['cache_size']} models loaded")
        print(f"Scalers Loaded: {data['scalers_loaded']}")
        
        print(f"\nModel Availability:")
        for dataset, models in data['models_status'].items():
            print(f"\n{dataset}:")
            for model_name, status in models.items():
                exists = "✅" if status['exists'] else "❌"
                loaded = "🔥" if status['loaded'] else "💤"
                print(f"  {exists} {loaded} {model_name:20s} | Type: {status['type']}")
        
        return True
    else:
        print(f"❌ Failed: {response.status_code}")
        return False

def run_all_tests():
    """Run all API tests"""
    print("\n" + "🚀"*40)
    print(" "*30 + "EXOPLANET API TEST SUITE")
    print("🚀"*40 + "\n")
    
    results = []
    
    # Test 1: API Info
    results.append(("API Info", test_api_info()))
    
    if not results[0][1]:
        print("\n❌ API is not running. Please start the server first:")
        print("   python main.py")
        print("   or")
        print("   uvicorn main:app --reload")
        return
    
    # Test 2: Dataset Features
    results.append(("KEPLER Features", test_dataset_features("KEPLER")))
    results.append(("K2 Features", test_dataset_features("K2")))
    
    # Test 3: Single Model Predictions
    results.append(("Single Model (XGBoost/KEPLER)", test_single_model_prediction("KEPLER", "XGBoost", KEPLER_SAMPLE)))
    results.append(("Single Model (Random_Forest/K2)", test_single_model_prediction("K2", "Random_Forest", K2_SAMPLE)))
    
    # Test 4: All Models Predictions
    results.append(("All Models (KEPLER)", test_all_models_prediction("KEPLER", KEPLER_SAMPLE)))
    results.append(("All Models (K2)", test_all_models_prediction("K2", K2_SAMPLE)))
    
    # Test 5: Models Status
    results.append(("Models Status", test_models_status()))
    
    # Summary
    print_header("TEST SUMMARY")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    print(f"{'='*60}")
    
    if passed == total:
        print("\n🎉 All tests passed! Your API is working perfectly!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check the errors above.")

if __name__ == "__main__":
    try:
        run_all_tests()
    except requests.exceptions.ConnectionError:
        print("\n❌ Cannot connect to API at http://localhost:8000")
        print("\nPlease make sure the FastAPI server is running:")
        print("  python main.py")
        print("  or")
        print("  uvicorn main:app --reload")
    except KeyboardInterrupt:
        print("\n\n⏹️  Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")