import requests
import json
import time

def test_api():
    """Test the Wine Classifier API"""
    base_url = "http://localhost:5001"
    
    print("=== WINE CLASSIFIER API TESTING ===")
    
    # Test 1: Health check
    print("\n1. Testing Health Check...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False
    
    # Test 2: API Info
    print("\n2. Testing API Info...")
    try:
        response = requests.get(f"{base_url}/info", timeout=10)
        if response.status_code == 200:
            print("✅ API info endpoint working")
            info = response.json()
            print(f"   Service: {info.get('service')}")
            print(f"   Model: {info.get('model')}")
            print(f"   Features: {len(info.get('features', []))}")
        else:
            print(f"❌ API info failed: {response.status_code}")
    except Exception as e:
        print(f"❌ API info error: {e}")
    
    # Test 3: Single prediction
    print("\n3. Testing Single Prediction...")
    test_sample = {
        "alcohol": 13.0,
        "malic_acid": 2.0,
        "ash": 2.5,
        "alcalinity_of_ash": 20.0,
        "magnesium": 100.0,
        "total_phenols": 2.5,
        "flavanoids": 2.0,
        "nonflavanoid_phenols": 0.3,
        "proanthocyanins": 1.5,
        "color_intensity": 5.0,
        "hue": 1.0,
        "od280/od315_of_diluted_wines": 2.5,
        "proline": 800.0,
        "location": 0
    }
    
    try:
        response = requests.post(
            f"{base_url}/predict",
            json=test_sample,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Single prediction successful")
            print(f"   Prediction: {result.get('prediction')}")
            print(f"   Wine Class: {result.get('wine_class')}")
            print(f"   Probability: {result.get('probability', 0):.4f}")
        else:
            print(f"❌ Single prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Single prediction error: {e}")
        return False
    
    # Test 4: Batch prediction
    print("\n4. Testing Batch Prediction...")
    batch_data = [test_sample, test_sample.copy()]
    batch_data[1]["alcohol"] = 12.5  # Slightly different sample
    
    try:
        response = requests.post(
            f"{base_url}/predict/batch",
            json=batch_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Batch prediction successful")
            print(f"   Total samples: {result.get('total_samples')}")
            print(f"   Predictions: {len(result.get('predictions', []))}")
            for pred in result.get('predictions', [])[:2]:  # Show first 2
                print(f"   Sample {pred.get('sample_index')}: {pred.get('wine_class')} ({pred.get('probability', 0):.4f})")
        else:
            print(f"❌ Batch prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Batch prediction error: {e}")
    
    print("\n=== API TESTING COMPLETED ===")
    return True

if __name__ == "__main__":
    # Wait a moment for API to be ready
    print("Waiting for API to be ready...")
    time.sleep(3)
    test_api()
