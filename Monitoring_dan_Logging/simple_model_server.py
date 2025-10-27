import flask
import joblib
import pandas as pd
import os
import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", category=UserWarning)


app = flask.Flask(__name__)# Dapatkan direktori tempat script ini berada
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Cari model.pkl di direktori yang sama
MODEL_FILENAME = os.path.join(SCRIPT_DIR, "model.pkl")

try:
    print(f"Mencoba memuat model dari: {MODEL_FILENAME}")
    if not os.path.exists(MODEL_FILENAME):
        raise FileNotFoundError(f"File {MODEL_FILENAME} tidak ditemukan di direktori ini.")
    model = joblib.load(MODEL_FILENAME)
    print("Model berhasil dimuat.")
except Exception as e:
    print(f"GAGAL memuat model: {e}")
    # Jika gagal, server tetap jalan tapi endpoint predict akan error
# -----------------

@app.route('/predict', methods=['POST'])
def predict():
    global model
    if model is None:
        return flask.jsonify({"error": "Model tidak berhasil dimuat saat startup."}), 500

    try:
        data = flask.request.get_json()

        # Asumsi input adalah format yang sama dengan inference.py: {"inputs": [[feat1, feat2, ...]]}
        # Kita perlu mengekstrak list fitur nya
        input_features = data.get('inputs')
        if not input_features or not isinstance(input_features, list) or not isinstance(input_features[0], list):
             raise ValueError("Format input tidak valid. Harusnya: {'inputs': [[val1, val2, ...]]}")

        # Lakukan prediksi (model.predict menerima array 2D)
        predictions = model.predict(input_features)

        # Ubah numpy array ke list Python biasa agar bisa di-JSON-kan
        result = {"predictions": predictions.tolist()}
        return flask.jsonify(result)

    except Exception as e:
        return flask.jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    print("Menjalankan server Flask sederhana di http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
