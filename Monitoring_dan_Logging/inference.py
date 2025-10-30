import requests
import time
import random
import pandas as pd
import os
import sys

# --- Konfigurasi ---
DATA_PATH = "../Membangun_model/PCOS_preprocessing.csv" 
EXPORTER_URL = "http://localhost:5001/predict"
# ------------------

def load_real_data(path):
    """Memuat data fitur asli dari CSV."""
    try:
        if not os.path.exists(path):
            print(f"Error: File data tidak ditemukan di {path}", file=sys.stderr)
            print("Pastikan path DATA_PATH di inference.py sudah benar.", file=sys.stderr)
            return None
        
        df = pd.read_csv(path)
        # Hapus kolom target
        if 'PCOS (Y/N)' in df.columns:
            df = df.drop('PCOS (Y/N)', axis=1)
        
        # Handle NaN/Inf jika ada (sama seperti di training)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        if df.isnull().sum().sum() > 0:
            print("Menemukan NaN di data, mengisi dengan 0 (ganti dengan mean jika perlu).")
            df.fillna(0, inplace=True) # Isi NaN dengan 0 (atau df.mean())

        # Ubah dataframe fitur menjadi list of lists
        return df.values.tolist()

    except Exception as e:
        print(f"Error saat memuat data: {e}", file=sys.stderr)
        return None

# --- Main ---
print("Memuat data inferensi asli...")
feature_list = load_real_data(DATA_PATH)
if feature_list is None:
    sys.exit(1)

print(f"Data berhasil dimuat. {len(feature_list)} sampel ditemukan.")
print(f"Memulai pengiriman inferensi ke: {EXPORTER_URL}")
print("Tekan Ctrl+C untuk berhenti.")

while True:
    try:
        # Pilih satu baris acak dari data asli
        random_sample = random.choice(feature_list)
        
        # Format payload sesuai dengan yang diharapkan 'prometheus_exporter.py'
        payload = {
            "inputs": [random_sample] 
        }

        response = requests.post(EXPORTER_URL, json=payload)

        if response.status_code == 200:
            print(f"Data: {random_sample[:4]}... -> Respon: {response.json()} - Sukses")
        else:
            print(f"Error: {response.status_code} - {response.text}")
            
        time.sleep(random.uniform(1.0, 3.0))

    except requests.exceptions.ConnectionError:
        print("Koneksi gagal. Pastikan stack (docker-compose & exporter) sudah berjalan.")
        time.sleep(5)
    except KeyboardInterrupt:
        print("\nInferensi dihentikan.")
        break