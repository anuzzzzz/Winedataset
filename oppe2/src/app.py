from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os
import traceback
import sys
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.instrumentation.flask import FlaskInstrumentor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize OpenTelemetry (basic setup)
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Create Flask app
app = Flask(__name__)

# Instrument Flask app with OpenTelemetry
FlaskInstrumentor().instrument_app(app)

# Global variables for models and scaler
model = None
feature_names = None

def load_models():
    """Load the trained models and preprocessing objects"""
    global model, feature_names
    
    try:
        logger.info("Starting model loading process...")
        
        # Print current working directory and contents
        current_dir = os.getcwd()
        logger.info(f"Current working directory: {current_dir}")
        logger.info(f"Directory contents: {os.listdir(current_dir)}")
        
        # Check models directory
        models_dir = "models"
        if os.path.exists(models_dir):
            logger.info(f"Models directory contents: {os.listdir(models_dir)}")
        else:
            logger.error(f"Models directory '{models_dir}' does not exist!")
            return False
        
        # Check data directory
        data_dir = "data"
        if os.path.exists(data_dir):
            logger.info(f"Data directory contents: {os.listdir(data_dir)}")
        else:
            logger.error(f"Data directory '{data_dir}' does not exist!")
        
        # Try to load model
        model_files = [
            "models/random_forest_model.pkl",
            "random_forest_model.pkl",
            "models/logistic_regression_model.pkl",
            "logistic_regression_model.pkl"
        ]
        
        model_loaded = False
        for model_path in model_files:
            if os.path.exists(model_path):
                try:
                    model = joblib.load(model_path)
                    logger.info(f"✅ Model loaded successfully from {model_path}")
                    model_loaded = True
                    break
                except Exception as e:
                    logger.warning(f"Failed to load model from {model_path}: {e}")
        
        if not model_loaded:
            logger.error("❌ No model could be loaded!")
            return False
        
        # Load feature names - try multiple sources
        feature_names = None
        
        # Try loading from data file
        data_files = ["data/wine_features.csv", "wine_features.csv"]
        for data_file in data_files:
            if os.path.exists(data_file):
                try:
                    features_df = pd.read_csv(data_file)
                    feature_names = features_df.columns.tolist()
                    logger.info(f"✅ Feature names loaded from {data_file}: {len(feature_names)} features")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load features from {data_file}: {e}")
        
        # Fallback to hardcoded feature names
        if feature_names is None:
            logger.info("Using fallback feature names...")
            from sklearn.datasets import load_wine
            wine_data = load_wine()
            feature_names = wine_data.feature_names.tolist() + ['location']
            logger.info(f"✅ Fallback feature names: {len(feature_names)} features")
        
        logger.info(f"Final feature names: {feature_names}")
        logger.info("✅ Model loading completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Critical error during model loading: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    with tracer.start_as_current_span("health_check"):
        status_info = {
            "service": "wine-classifier",
            "model_loaded": model is not None,
            "features_count": len(feature_names) if feature_names else 0,
            "working_dir": os.getcwd(),
            "python_version": sys.version
        }
        
        if model is not None:
            status_info["status"] = "healthy"
            return jsonify(status_info)
        else:
            status_info["status"] = "unhealthy"
            status_info["error"] = "Model not loaded"
            # Try to reload models
            logger.info("Health check failed, attempting to reload models...")
            if load_models():
                status_info["status"] = "healthy"
                status_info["model_loaded"] = True
                return jsonify(status_info)
            return jsonify(status_info), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Single prediction endpoint"""
    with tracer.start_as_current_span("single_prediction"):
        try:
            # Check if model is loaded
            if model is None:
                logger.warning("Model not loaded, attempting to reload...")
                if not load_models():
                    return jsonify({"error": "Model not loaded and reload failed"}), 500
            
            # Get input data
            data = request.json
            if not data:
                return jsonify({"error": "No input data provided"}), 400
            
            logger.info(f"Received prediction request with keys: {list(data.keys())}")
            
            # Convert to DataFrame
            df = pd.DataFrame([data])
            
            # Add missing features with defaults if needed
            for feature in feature_names:
                if feature not in df.columns:
                    if feature == 'location':
                        df[feature] = 0  # Default location
                    else:
                        logger.warning(f"Missing feature {feature}, setting to 0")
                        df[feature] = 0  # Default to 0 instead of error
            
            # Reorder columns to match training data
            df = df[feature_names]
            
            # Make prediction
            prediction = model.predict(df)[0]
            probabilities = model.predict_proba(df)[0]
            max_probability = float(probabilities.max())
            
            # Wine cultivar names
            wine_classes = ["Cultivar 0", "Cultivar 1", "Cultivar 2"]
            
            result = {
                "prediction": int(prediction),
                "probability": max_probability,
                "wine_class": wine_classes[prediction],
                "all_probabilities": {
                    wine_classes[i]: float(prob) for i, prob in enumerate(probabilities)
                }
            }
            
            logger.info(f"Prediction successful: {result['wine_class']} ({max_probability:.3f})")
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint"""
    with tracer.start_as_current_span("batch_prediction"):
        try:
            if model is None:
                if not load_models():
                    return jsonify({"error": "Model not loaded"}), 500
            
            data = request.json
            if not data or not isinstance(data, list):
                return jsonify({"error": "Input should be a list of samples"}), 400
            
            logger.info(f"Received batch prediction request with {len(data)} samples")
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Add missing features with defaults
            for feature in feature_names:
                if feature not in df.columns:
                    df[feature] = 0
            
            # Reorder columns
            df = df[feature_names]
            
            # Make predictions
            predictions = model.predict(df)
            probabilities = model.predict_proba(df)
            
            wine_classes = ["Cultivar 0", "Cultivar 1", "Cultivar 2"]
            
            results = []
            for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
                results.append({
                    "sample_index": i,
                    "prediction": int(pred),
                    "probability": float(probs.max()),
                    "wine_class": wine_classes[pred]
                })
            
            response = {
                "predictions": results,
                "total_samples": len(data)
            }
            
            logger.info(f"Batch prediction completed: {len(results)} results")
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"Batch prediction error: {str(e)}")
            return jsonify({"error": f"Batch prediction failed: {str(e)}"}), 500

@app.route('/info', methods=['GET'])
def info():
    """API information endpoint"""
    return jsonify({
        "service": "Wine Classifier API",
        "version": "1.0.0",
        "model": "Random Forest Classifier",
        "features": feature_names if feature_names else [],
        "model_loaded": model is not None,
        "endpoints": {
            "/health": "Health check",
            "/predict": "Single prediction (POST)",
            "/predict/batch": "Batch prediction (POST)",
            "/info": "API information"
        }
    })

# Initialize the app
if __name__ == '__main__':
    logger.info("Starting Wine Classifier API...")
    
    # Load models
    if load_models():
        logger.info("🍷 Wine Classifier API ready to serve!")
        app.run(host='0.0.0.0', port=5001, debug=False)
    else:
        logger.error("❌ Failed to load models initially, but API will still start.")
        logger.error("Models will be loaded on first request.")
        app.run(host='0.0.0.0', port=5001, debug=False)
