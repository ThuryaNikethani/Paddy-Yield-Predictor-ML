"""
Smart Paddy Monitoring and Yield Risk Prediction Application
Flask Backend API for disease classification and yield prediction
Integrates data from: CSV datasets, RRDI PDF disease profiles, Yala/Maha production PDFs,
meteorological PDFs, and Helgi chart metadata
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import joblib
import cv2
from PIL import Image
import io
import os
import json
from datetime import datetime
import pandas as pd

from pdf_data import (
    DISEASE_KNOWLEDGE, YALA_2024_DATA, MAHA_2024_2025_DATA,
    STATION_TEMPERATURE, STATION_RAINFALL, STATION_HUMIDITY,
    DISTRICT_TO_STATION, HELGI_METADATA,
    get_district_yield, get_district_weather, get_all_districts
)

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
MODELS_FOLDER = 'models'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

# Disease classes (CNN-classifiable from image dataset)
# NOTE: The 'Leafsmut' folder actually contains Rice Blast images (BLAST1_*.jpg)
# Verified by matching RRDI PDF symptom descriptions to actual image content
DISEASE_CLASSES = ['Bacterial Leaf Blight', 'Brown Spot', 'Rice Blast']

# Build comprehensive disease information dynamically from RRDI PDF data
# This ensures app.py always reflects the exact content from pdf_data.py
def _build_disease_info():
    """Build DISEASE_INFO from DISEASE_KNOWLEDGE (pdf_data.py) — exact RRDI PDF content"""
    info = {}
    for disease_name, dk in DISEASE_KNOWLEDGE.items():
        # Combine within-season + next-season management into single treatment list
        all_treatments = list(dk.get('management', []))
        next_season = dk.get('next_season_management', [])
        if next_season:
            all_treatments.append('--- For next season ---')
            all_treatments.extend(next_season)

        info[disease_name] = {
            'description': f"{disease_name} — caused by {dk.get('causative_agent', 'Unknown')}. "
                           f"Affects: {dk.get('affected_parts', 'N/A')}. "
                           f"Stage: {dk.get('affected_stages', 'N/A')}. "
                           f"(Source: RRDI, Dept. of Agriculture Sri Lanka)",
            'causative_agent': dk.get('causative_agent', ''),
            'affected_parts': dk.get('affected_parts', ''),
            'affected_stages': dk.get('affected_stages', ''),
            'severity_note': dk.get('severity_note', ''),
            'symptoms': '; '.join(dk.get('symptoms', [])),
            'symptoms_list': dk.get('symptoms', []),
            'favorable_conditions': dk.get('favorable_conditions', {}),
            'treatment': all_treatments,
            'resistant_varieties': dk.get('resistant_varieties', []),
            'susceptible_varieties': dk.get('susceptible_varieties', []),
            'source_pdf': dk.get('source', ''),
            'url': dk.get('url', ''),
        }
    return info

DISEASE_INFO = _build_disease_info()

# Load models
print("Loading models...")
try:
    disease_model = tf.keras.models.load_model(os.path.join(MODELS_FOLDER, 'disease_classification_model.h5'))
    yield_model = joblib.load(os.path.join(MODELS_FOLDER, 'yield_prediction_model.pkl'))
    loss_model = joblib.load(os.path.join(MODELS_FOLDER, 'yield_loss_model.pkl'))
    scaler = joblib.load(os.path.join(MODELS_FOLDER, 'feature_scaler.pkl'))
    season_encoder = joblib.load(os.path.join(MODELS_FOLDER, 'season_encoder.pkl'))
    feature_cols = joblib.load(os.path.join(MODELS_FOLDER, 'feature_columns.pkl'))
    print("✓ All models loaded successfully")
except Exception as e:
    print(f"⚠ Warning: Could not load models - {e}")
    print("  Please train models first using train_disease_model.py and train_yield_model.py")
    disease_model = None
    yield_model = None

# Load district yield reference from PDFs (if available)
district_yield_ref = {}
try:
    ref_path = os.path.join(MODELS_FOLDER, 'district_yield_reference.json')
    if os.path.exists(ref_path):
        with open(ref_path, 'r') as f:
            district_yield_ref = json.load(f)
        print(f"✓ District yield reference loaded ({len(district_yield_ref)} districts from PDFs)")
except Exception as e:
    print(f"⚠ Could not load district yield reference: {e}")

# Load location data
location_df = pd.read_csv('locationData.csv')

@app.route('/')
def index():
    """Serve main application page"""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': disease_model is not None and yield_model is not None
    })

@app.route('/api/locations', methods=['GET'])
def get_locations():
    """Get available locations"""
    locations = location_df[['location_id', 'city_name', 'latitude', 'longitude']].to_dict('records')
    return jsonify({'locations': locations})

def preprocess_image(image_data):
    """Preprocess image for disease classification"""
    try:
        # Load image
        img = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to model input size
        img = img.resize((224, 224))
        
        # Convert to array and normalize
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {str(e)}")

def classify_disease(img_array):
    """Classify disease from image"""
    if disease_model is None:
        raise ValueError("Disease model not loaded")
    
    # Predict
    predictions = disease_model.predict(img_array)
    confidence_scores = predictions[0]
    
    # Get top prediction
    predicted_class_idx = np.argmax(confidence_scores)
    confidence = float(confidence_scores[predicted_class_idx]) * 100
    disease_name = DISEASE_CLASSES[predicted_class_idx]
    
    # Calculate severity based on confidence and pattern
    if confidence > 90:
        severity = 'High'
        severity_level = 3
    elif confidence > 75:
        severity = 'Medium'
        severity_level = 2
    elif confidence > 60:
        severity = 'Low'
        severity_level = 1
    else:
        severity = 'Very Low'
        severity_level = 0
    
    return {
        'disease': disease_name,
        'confidence': round(confidence, 2),
        'severity': severity,
        'severity_level': severity_level,
        'all_predictions': {
            DISEASE_CLASSES[i]: round(float(confidence_scores[i]) * 100, 2)
            for i in range(len(DISEASE_CLASSES))
        }
    }

def predict_yield(location_id, season, disease_severity, temperature=None, rainfall=None):
    """Predict yield and loss"""
    if yield_model is None or loss_model is None:
        raise ValueError("Yield models not loaded")
    
    # Get location data
    loc_data = location_df[location_df['location_id'] == location_id].iloc[0]
    
    # Default weather values if not provided
    if temperature is None:
        temperature = 26.5  # Average for Sri Lanka
    if rainfall is None:
        rainfall = 1500  # Average annual rainfall
    
    # Encode season
    season_encoded = season_encoder.transform([season])[0]
    
    # Prepare features
    features = np.array([[
        location_id,
        season_encoded,
        temperature,
        temperature + 4,  # max
        temperature - 4,  # min
        rainfall,
        rainfall / 12,  # monthly average
        4.5,  # evapotranspiration
        10.0,  # wind speed
        disease_severity,
        float(loc_data['elevation']),
        float(loc_data['latitude']),
        float(loc_data['longitude'])
    ]])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict
    predicted_yield = yield_model.predict(features_scaled)[0]
    predicted_loss = loss_model.predict(features_scaled)[0]
    
    # Ensure realistic values
    predicted_yield = max(1.0, min(8.0, predicted_yield))
    predicted_loss = max(0, min(100, predicted_loss))
    
    return {
        'predicted_yield': round(float(predicted_yield), 2),
        'yield_loss_percentage': round(float(predicted_loss), 2)
    }

def calculate_risk_level(disease_severity, confidence, yield_loss):
    """Calculate overall risk level"""
    risk_score = (disease_severity * 25) + (yield_loss * 0.5) + ((100 - confidence) * 0.25)
    
    if risk_score > 60:
        return 'High', '#dc3545'
    elif risk_score > 30:
        return 'Medium', '#ffc107'
    else:
        return 'Low', '#28a745'

@app.route('/api/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        # Validate request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Get parameters
        location_id = int(request.form.get('location_id', 0))
        season = request.form.get('season', 'Maha')
        temperature = float(request.form.get('temperature', 26.5))
        rainfall = float(request.form.get('rainfall', 1500))
        
        # Validate parameters
        if season not in ['Maha', 'Yala']:
            return jsonify({'error': 'Season must be either Maha or Yala'}), 400
        
        # Read image
        image_data = image_file.read()
        
        # Preprocess image
        img_array = preprocess_image(image_data)
        
        # Classify disease
        disease_result = classify_disease(img_array)
        
        # Predict yield
        yield_result = predict_yield(
            location_id,
            season,
            disease_result['severity_level'],
            temperature,
            rainfall
        )
        
        # Calculate risk level
        risk_level, risk_color = calculate_risk_level(
            disease_result['severity_level'],
            disease_result['confidence'],
            yield_result['yield_loss_percentage']
        )
        
        # Get disease info (comprehensive from RRDI PDFs)
        disease_name = disease_result['disease']
        disease_details = DISEASE_INFO.get(disease_name, {})
        
        # Get district yield reference if available
        district_ref = {}
        if district_yield_ref:
            for dist_name, dist_data in district_yield_ref.items():
                district_ref = dist_data
                break
        
        # Prepare response
        response = {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'disease_detection': {
                'disease_name': disease_name,
                'confidence': disease_result['confidence'],
                'severity': disease_result['severity'],
                'severity_level': disease_result['severity_level'],
                'all_predictions': disease_result['all_predictions'],
                'description': disease_details.get('description', ''),
                'causative_agent': disease_details.get('causative_agent', ''),
                'affected_parts': disease_details.get('affected_parts', ''),
                'affected_stages': disease_details.get('affected_stages', ''),
                'severity_note': disease_details.get('severity_note', ''),
                'symptoms': disease_details.get('symptoms', ''),
                'symptoms_list': disease_details.get('symptoms_list', []),
                'favorable_conditions': disease_details.get('favorable_conditions', {}),
                'treatment': disease_details.get('treatment', []),
                'resistant_varieties': disease_details.get('resistant_varieties', []),
                'susceptible_varieties': disease_details.get('susceptible_varieties', []),
                'source': disease_details.get('source_pdf', 'RRDI PDF Disease Profile'),
                'url': disease_details.get('url', ''),
            },
            'yield_prediction': {
                'predicted_yield_tons_per_ha': yield_result['predicted_yield'],
                'yield_loss_percentage': yield_result['yield_loss_percentage'],
                'expected_production_per_acre': round(yield_result['predicted_yield'] * 0.4047, 2)
            },
            'risk_assessment': {
                'overall_risk_level': risk_level,
                'risk_color': risk_color,
                'recommendation': get_recommendation(risk_level, disease_name)
            },
            'environmental_data': {
                'location_id': location_id,
                'season': season,
                'temperature': temperature,
                'rainfall': rainfall
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def get_recommendation(risk_level, disease_name):
    """Get recommendation based on risk level"""
    disease_detail = DISEASE_INFO.get(disease_name, {})
    conditions = disease_detail.get('favorable_conditions', {})
    conditions_str = ""
    if conditions:
        temp = conditions.get('temperature', '')
        humid = conditions.get('humidity', '')
        if temp:
            conditions_str += f" Favorable temp: {temp}."
        if humid:
            conditions_str += f" Favorable humidity: {humid}."
    
    recommendations = {
        'High': f"⚠️ URGENT ACTION REQUIRED: Severe {disease_name} detected (caused by {disease_detail.get('causative_agent', 'unknown pathogen')}). Immediate treatment necessary.{conditions_str} Apply recommended treatments immediately. Consult nearest RRDI extension officer.",
        'Medium': f"⚡ ATTENTION NEEDED: Moderate {disease_name} infection detected.{conditions_str} Apply treatment measures soon. Monitor closely and maintain proper field management.",
        'Low': f"✓ MONITOR REGULARLY: Early stage {disease_name} detected. Implement preventive measures.{conditions_str} Continue normal cultivation practices with close monitoring."
    }
    return recommendations.get(risk_level, "Continue regular monitoring and maintenance")

@app.route('/api/diseases', methods=['GET'])
def get_all_disease_info():
    """Get complete disease knowledge from RRDI PDFs (all 8 diseases documented)"""
    return jsonify({
        'total_diseases': len(DISEASE_INFO),
        'cnn_classifiable': DISEASE_CLASSES,
        'rrdi_documented': [d for d in DISEASE_KNOWLEDGE.keys() if DISEASE_KNOWLEDGE[d].get('url')],
        'knowledge_only': [d for d in DISEASE_INFO if d not in DISEASE_CLASSES],
        'diseases': DISEASE_INFO,
        'source': 'Rice Research and Development Institute (RRDI), Department of Agriculture Sri Lanka',
        'note': 'Image dataset contains Bacterial Blight, Brown Spot, and Rice Blast. Other diseases are knowledge-only entries from RRDI PDFs.'
    })

@app.route('/api/disease/<disease_name>', methods=['GET'])
def get_disease_detail(disease_name):
    """Get detailed info for a specific disease"""
    # Try exact match first, then case-insensitive
    info = DISEASE_INFO.get(disease_name)
    if not info:
        for key, val in DISEASE_INFO.items():
            if key.lower() == disease_name.lower():
                info = val
                disease_name = key
                break
    if not info:
        return jsonify({'error': f'Disease "{disease_name}" not found. Available: {list(DISEASE_INFO.keys())}'}), 404
    return jsonify({'disease': disease_name, **info})

@app.route('/api/districts', methods=['GET'])
def get_districts():
    """Get all districts with yield data from production PDFs"""
    districts = get_all_districts()
    district_data = {}
    for d in districts:
        yield_info = get_district_yield(d)
        weather = get_district_weather(d)
        district_data[d] = {
            'yield': yield_info,
            'weather': weather
        }
    return jsonify({
        'total_districts': len(districts),
        'districts': district_data,
        'sources': ['Yala2024Metric.pdf', '2024_2025Maha_Metric.pdf', '1.3.pdf', '1.5.pdf', '1.6.pdf']
    })

@app.route('/api/district/<district_name>', methods=['GET'])
def get_district_detail(district_name):
    """Get detailed yield and weather data for a specific district"""
    yield_info = get_district_yield(district_name.upper())
    weather = get_district_weather(district_name.upper())
    if not yield_info.get('yala') and not yield_info.get('maha'):
        return jsonify({'error': f'District "{district_name}" not found. Available: {get_all_districts()}'}), 404
    return jsonify({
        'district': district_name.upper(),
        'yield_data': yield_info,
        'weather_data': weather,
        'sources': 'Yala/Maha production PDFs + meteorological PDFs'
    })

@app.route('/api/data-sources', methods=['GET'])
def get_data_sources():
    """Get summary of all data sources used"""
    return jsonify({
        'pdf_sources': {
            'production': {
                'yala_2024': f"{len(YALA_2024_DATA['districts'])} districts, avg {YALA_2024_DATA['national_avg_yield_kg_ha']} kg/ha",
                'maha_2024_25': f"{len(MAHA_2024_2025_DATA['districts'])} districts, avg {MAHA_2024_2025_DATA['national_avg_yield_kg_ha']} kg/ha"
            },
            'meteorological': {
                'temperature_stations': len(STATION_TEMPERATURE),
                'rainfall_stations': len(STATION_RAINFALL),
                'humidity_stations': len(STATION_HUMIDITY)
            },
            'disease_profiles': f"{len(DISEASE_KNOWLEDGE)} diseases from RRDI"
        },
        'helgi_sources': {k: v['title'] for k, v in HELGI_METADATA.items()},
        'csv_sources': ['weatherData.csv', 'locationData.csv', 'UNdata_Export CSV']
    })

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("SMART PADDY MONITORING SYSTEM - Backend Server")
    print("=" * 60)
    print("\nServer starting...")
    print("• API endpoint: http://localhost:5000")
    print("• Health check: http://localhost:5000/api/health")
    print("• Web interface: http://localhost:5000")
    print("• Disease API: http://localhost:5000/api/diseases")
    print("• Districts API: http://localhost:5000/api/districts")
    print("• Data Sources: http://localhost:5000/api/data-sources")
    print(f"\nData Integration: {len(DISEASE_INFO)} diseases (all from RRDI PDFs)")
    print(f"Districts: {len(YALA_2024_DATA['districts'])} Yala + {len(MAHA_2024_2025_DATA['districts'])} Maha")
    print("\n" + "=" * 60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
