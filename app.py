"""
Flask API that serves Random Forest yield predictions.
Loads the trained model (random_forest_yield.pkl) and exposes /predict endpoint.
If the model file doesn't exist, it trains one on startup.
"""

import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

# ---------------------------------------------
# Load or train the model on startup
# ---------------------------------------------
MODEL_PATH = 'random_forest_yield.pkl'

if not os.path.exists(MODEL_PATH):
    print("Model not found. Training now...")
    import train_model  # runs train_model.py which saves the model

model = joblib.load(MODEL_PATH)
print("✅ Model loaded successfully.")

# ---------------------------------------------
# Create Flask app
# ---------------------------------------------
app = Flask(__name__)
CORS(app)  # allow requests from your Vercel dashboard


@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'service': 'Zimbabwe Farm ML API',
        'status': 'running',
        'endpoints': {
            'POST /predict': 'Predict maize yield',
            'GET /health': 'Check API health',
        },
        'model': 'Random Forest Regressor',
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_loaded': True})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        rainfall = float(data.get('rainfall', 700))
        temperature = float(data.get('temperature', 24))
        fertilizer = float(data.get('fertilizer', 60))

        features = np.array([[rainfall, temperature, fertilizer]])
        prediction = model.predict(features)[0]

        return jsonify({
            'success': True,
            'predicted_yield_t_ha': round(float(prediction), 3),
            'inputs': {
                'rainfall_mm': rainfall,
                'temperature_c': temperature,
                'fertilizer_kg_ha': fertilizer,
            },
            'confidence_range': {
                'low': round(float(prediction) - 0.25, 3),
                'high': round(float(prediction) + 0.25, 3),
            },
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
