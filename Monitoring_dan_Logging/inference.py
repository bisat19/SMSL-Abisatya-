import requests
import time
import random

EXPORTER_URL = "http://localhost:5001/predict"

def generate_sample_data():
    """Membuat satu sampel data acak dengan 10 fitur."""
    
    # 1. Follicle No. (R)
    follicle_r = random.randint(0, 20)
    # 2. Follicle No. (L)
    follicle_l = random.randint(0, 20)
    # 3. Skin darkening (Y/N)
    skin_darkening = random.choice([0, 1])
    # 4. hair growth(Y/N)
    hair_growth = random.choice([0, 1])
    # 5. Weight gain(Y/N)
    weight_gain = random.choice([0, 1])
    # 6. Cycle(R/I) (Asumsi 0=Regular, 1=Irregular)
    cycle_ri = random.choice([0, 1])
    # 7. Fast food (Y/N)
    fast_food = random.choice([0, 1])
    # 8. Cycle length(days)
    cycle_length = random.randint(2, 10)
    # 9. Age (yrs)
    age_yrs = random.randint(20, 45)
    # 10. Marraige Status (Yrs)
    marriage_yrs = random.randint(0, 20)
    
    # Kembalikan sebagai list dengan urutan yang benar
    return [
        follicle_r, follicle_l, skin_darkening, hair_growth,
        weight_gain, cycle_ri, fast_food, cycle_length,
        age_yrs, marriage_yrs
    ]

print(f"Memulai pengiriman inferensi ke: {EXPORTER_URL}")
print("Tekan Ctrl+C untuk berhenti.")

while True:
    try:
        sample_features = generate_sample_data()
        
        # Format payload sesuai dengan yang diharapkan 'prometheus_exporter.py'
        payload = {
            "inputs": [sample_features] 
        }

        response = requests.post(EXPORTER_URL, json=payload)

        if response.status_code == 200:
            print(f"Data: {sample_features} -> Respon: {response.json()} - Sukses")
        else:
            print(f"Error: {response.status_code} - {response.text}")
            
        time.sleep(random.uniform(1.0, 3.0))

    except requests.exceptions.ConnectionError:
        print("Koneksi gagal. Pastikan stack docker-compose sudah berjalan.")
        time.sleep(5)
    except KeyboardInterrupt:
        print("\nInferensi dihentikan.")
        break
