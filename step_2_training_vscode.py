import pandas as pd
from prophet import Prophet
import joblib
import os
import logging

# Menonaktifkan logging info dari Prophet agar tidak terlalu "berisik"
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

# --- KONFIGURASI ---

# 1. Lokasi file input (HASIL DARI STEP 1)
# Pastikan path ini benar
BASE_DIR = r"C:\Games\UPI Kampus Serang\Tugas\Semester 5\ASIK\WEBSITE"
INPUT_FILE_PATH = os.path.join(BASE_DIR, "data_bersih_model_ready.csv")

# 2. Lokasi file output (MODEL BARU KITA)
OUTPUT_MODEL_FILE = os.path.join(BASE_DIR, "prophet_models.pkl")

print(f"Memulai Step 2: Training Model Prophet...")

# --- 1. MEMUAT DATA BERSIH ---
try:
    # 'parse_dates' sangat penting agar Prophet membaca 'ds' sebagai tanggal
    df = pd.read_csv(INPUT_FILE_PATH, parse_dates=['ds'])
except FileNotFoundError:
    print(f"ERROR: File tidak ditemukan di {INPUT_FILE_PATH}")
    print("Pastikan Anda sudah menjalankan Step 1 dan file tersebut ada.")
    raise SystemExit
except Exception as e:
    print(f"Terjadi error saat membaca file: {e}")
    raise SystemExit

print(f"Data bersih dimuat: {len(df)} baris.")

# --- 2. MELATIH MODEL (LOOP PER IKAN) ---
print("Melatih model Prophet untuk setiap ikan...")

all_prophet_models = {} # Kamus untuk menyimpan semua model
unique_fish = df['nama_ikan'].unique()
model_count = 0

for ikan in unique_fish:
    # Ambil data untuk ikan ini (hanya kolom ds dan y)
    fish_data = df[df['nama_ikan'] == ikan][['ds', 'y']]
    
    # Cek apakah data cukup. 
    # Prophet butuh minimal 2 titik, tapi idealnya > 12 (1 tahun)
    if len(fish_data) < 12:
        print(f"  -> Melewatkan {ikan} (data tidak cukup, < 12 bulan)")
        continue
        
    print(f"  -> Melatih model untuk: {ikan} ({len(fish_data)} data poin)...")
    
    # Inisialisasi model Prophet
    m = Prophet(
        seasonality_mode='multiplicative', # Pola musimannya dikalikan tren
        yearly_seasonality=True,         # Paham pola tahunan
        weekly_seasonality=False,        # Tidak perlu pola mingguan
        daily_seasonality=False          # Tidak perlu pola harian
    )
    
    # Latih (fit) model ke data ikan ini
    try:
        m.fit(fish_data)
        
        # Simpan model yang sudah dilatih ke kamus
        all_prophet_models[ikan] = m
        model_count += 1
    except Exception as e:
        print(f"  -> GAGAL melatih {ikan}. Error: {e}")

print(f"\nTraining selesai.")
print(f"Berhasil melatih {model_count} dari {len(unique_fish)} ikan unik.")

# --- 3. SIMPAN KAMUS MODEL ---
if model_count > 0:
    print(f"Menyimpan kamus {model_count} model ke {OUTPUT_MODEL_FILE}...")
    try:
        joblib.dump(all_prophet_models, OUTPUT_MODEL_FILE)
        print(f"\nBERHASIL!")
        print(f"File model baru telah disimpan di:")
        print(f"{OUTPUT_MODEL_FILE}")
        print("\nLangkah selanjutnya (Step 3) adalah menjalankan 'step_3_app_prophet.py'.")
    except Exception as e:
        print(f"\nERROR: Gagal menyimpan file model. Detail: {e}")
else:
    print("Tidak ada model yang dilatih.")