import time
import requests
from flask import Flask, request, jsonify
from prometheus_client import Histogram, Counter, Gauge, make_wsgi_app
from werkzeug.middleware.dispatcher import DispatcherMiddleware

app = Flask(__name__)

# Metrik 1: Latency (Histogram)
REQUEST_LATENCY = Histogram(
    'pcos_request_latency_seconds',
    'Latensi request model PCOS (detik)',
    ['endpoint']
)

# Metrik 2: Request Count (Counter)
REQUEST_COUNT = Counter(
    'pcos_request_count',
    'Total request model PCOS',
    ['endpoint', 'method', 'status_code']
)

# Metrik 3: Prediction Result (Counter)
PREDICTION_RESULT = Counter(
    'pcos_prediction_result_total',
    'Total hasil prediksi (Ya/Tidak)',
    ['class_name'] # PCOS_YES atau PCOS_NO
)

# Metrik 4-13: Metrik Fitur Spesifik (10 Fitur = 10 Metriks)
FEATURE_GAUGES = {
    'follicle_r': Gauge('pcos_input_follicle_r', 'Nilai input Follicle No. (R)'),
    'follicle_l': Gauge('pcos_input_follicle_l', 'Nilai input Follicle No. (L)'),
    'cycle_length': Gauge('pcos_input_cycle_length_days', 'Nilai input Cycle length(days)'),
    'age_yrs': Gauge('pcos_input_age_yrs', 'Nilai input Age (yrs)'),
    'marriage_yrs': Gauge('pcos_input_marriage_status_yrs', 'Nilai input Marraige Status (Yrs)')
}

FEATURE_COUNTERS = {
    'skin_darkening': Counter('pcos_input_skin_darkening_total', 'Total input Skin darkening (Y/N)', ['value']),
    'hair_growth': Counter('pcos_input_hair_growth_total', 'Total input hair growth(Y/N)', ['value']),
    'weight_gain': Counter('pcos_input_weight_gain_total', 'Total input Weight gain(Y/N)', ['value']),
    'cycle_ri': Counter('pcos_input_cycle_ri_total', 'Total input Cycle(R/I)', ['value']),
    'fast_food': Counter('pcos_input_fast_food_total', 'Total input Fast food (Y/N)', ['value'])
}

# Metrik 14: Health Check Model Server (Gauge)
MODEL_SERVER_UP = Gauge(
    'pcos_model_server_up',
    'Status Model Server (1=UP, 0=DOWN)'
)
# ------------------------------------------------------------------

# Endpoint untuk Prometheus
app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
    '/metrics': make_wsgi_app()
})

# Endpoint Health Check sederhana untuk service ini
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    try:
        data = request.get_json()
        payload = {"inputs": data['inputs']}
        response = requests.post(
            "http://localhost:5000/predict", 
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        response.raise_for_status() # Error jika status code bukan 2xx
        result = response.json()
        MODEL_SERVER_UP.set(1) # Tandai model server UP

        # --- Update Metriks ---
        latency = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint='/predict').observe(latency)
        REQUEST_COUNT.labels(endpoint='/predict', method='POST', status_code=200).inc()

        # Update Metrik Prediksi
        prediction = result['predictions'][0] 
        class_name = 'PCOS_YES' if prediction == 1 else 'PCOS_NO'
        PREDICTION_RESULT.labels(class_name=class_name).inc()
        
        # Update Metrik Fitur
        input_features = data['inputs'][0] # Ambil list 10 fitur
        
        # Set Gauges (untuk nilai numerik)
        FEATURE_GAUGES['follicle_r'].set(input_features[0])
        FEATURE_GAUGES['follicle_l'].set(input_features[1])
        FEATURE_GAUGES['cycle_length'].set(input_features[7])
        FEATURE_GAUGES['age_yrs'].set(input_features[8])
        FEATURE_GAUGES['marriage_yrs'].set(input_features[9])
        
        # Inc Counters (untuk nilai kategorikal Y/N atau R/I)
        FEATURE_COUNTERS['skin_darkening'].labels(value=input_features[2]).inc()
        FEATURE_COUNTERS['hair_growth'].labels(value=input_features[3]).inc()
        FEATURE_COUNTERS['weight_gain'].labels(value=input_features[4]).inc()
        FEATURE_COUNTERS['cycle_ri'].labels(value=input_features[5]).inc()
        FEATURE_COUNTERS['fast_food'].labels(value=input_features[6]).inc()
        # ------------------------

        return jsonify(result)

    except Exception as e:
        MODEL_SERVER_UP.set(0) # Tandai model server DOWN
        
        latency = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint='/predict').observe(latency)
        REQUEST_COUNT.labels(endpoint='/predict', method='POST', status_code=500).inc()
        
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
