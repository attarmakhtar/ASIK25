import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import numpy as np
import os

# --- KONFIGURASI ---

# 1. Daftar 6 file CSV yang Anda unggah
file_list = [
    "Data Produksi Perikanan Tangkap 2019-2024.xlsx - 2019.csv",
    "Data Produksi Perikanan Tangkap 2019-2024.xlsx - 2020.csv",
    "Data Produksi Perikanan Tangkap 2019-2024.xlsx - 2021.csv",
    "Data Produksi Perikanan Tangkap 2019-2024.xlsx - 2022.csv",
    "Data Produksi Perikanan Tangkap 2019-2024.xlsx - 2023.csv",
    "Data Produksi Perikanan Tangkap 2019-2024.xlsx - 2024.csv"
]

# 2. Nama file model output (akan menimpa model lama)
OUTPUT_MODEL_FILE = 'time_series_models.pkl'

print("Memulai proses training model final...")

all_data = []

# --- TAHAP 1: BACA DAN GABUNGKAN 6 FILE CSV ---

print("Membaca dan menggabungkan 6 file CSV...")
for file in file_list:
    if not os.path.exists(file):
        print(f"PERINGATAN: File {file} tidak ditemukan. Dilewati.")
        continue
    
    # Ekstrak tahun dari nama file (mengambil 4 angka sebelum ".csv")
    try:
        year = int(file.split(' - ')[-1].split('.')[0])
    except:
        print(f"Gagal mengekstrak tahun dari nama file: {file}")
        continue
        
    df = pd.read_csv(file)
    
    # Bersihkan nama kolom (menghapus spasi, membuat huruf kecil)
    df.columns = df.columns.str.strip().str.lower()
    
    # Cari kolom 'nama ikan' dan 'jumlah (kg)'
    # (Ini menangani jika nama kolomnya 'Jumlah (Kg)' atau 'jumlah (kg)')
    
    col_nama = None
    col_jumlah = None

    if 'nama ikan' in df.columns: col_nama = 'nama ikan'
    if 'jumlah (kg)' in df.columns: col_jumlah = 'jumlah (kg)'
        
    if not col_nama or not col_jumlah:
        print(f"PERINGATAN: File {file} tidak memiliki kolom 'nama ikan' or 'jumlah (kg)'. Dilewati.")
        continue
        
    # Ambil data yang kita butuhkan
    temp_df = df[[col_nama, col_jumlah]].copy()
    temp_df['tahun'] = year
    all_data.append(temp_df)

if not all_data:
    print("ERROR: Tidak ada data yang berhasil dibaca. Pastikan file CSV ada dan formatnya benar.")
    exit()

# Gabungkan semua data menjadi satu DataFrame
df_aggregated = pd.concat(all_data, ignore_index=True)

# Ganti nama kolom agar konsisten
df_aggregated.rename(columns={'nama ikan': 'nama_ikan', 'jumlah (kg)': 'total_kg'}, inplace=True)

# Bersihkan data
df_aggregated['nama_ikan'] = df_aggregated['nama_ikan'].str.strip()
# Konversi total_kg ke angka, paksa error menjadi 0
df_aggregated['total_kg'] = pd.to_numeric(df_aggregated['total_kg'], errors='coerce').fillna(0)

print(f"Total {len(df_aggregated)} data tahunan ikan berhasil digabungkan.")

# --- TAHAP 2: LATIH MODEL (SAMA SEPERTI SEBELUMNYA) ---
print("Melatih model Regresi Linear untuk setiap ikan...")

all_fish_models = {} 
unique_fish = df_aggregated['nama_ikan'].unique()
model_count = 0

for ikan in unique_fish:
    fish_data = df_aggregated[df_aggregated['nama_ikan'] == ikan].sort_values('tahun')
    
    # Butuh minimal 2 tahun data untuk membuat garis tren
    if len(fish_data) >= 2:
        X = fish_data[['tahun']] 
        y = fish_data['total_kg']
        
        model = LinearRegression()
        model.fit(X, y)
        
        all_fish_models[ikan] = model 
        model_count += 1
        
print(f"Berhasil melatih {model_count} model ikan.")

# --- TAHAP 3: SIMPAN MODEL ---
if model_count > 0:
    joblib.dump(all_fish_models, OUTPUT_MODEL_FILE)
    print(f"Kamus model berhasil DISIMPAN/DITIMPA ke {OUTPUT_MODEL_FILE}")
    print("\nTraining selesai. Anda sekarang bisa menjalankan 'app.py'.")
else:
    print("Tidak ada model yang dilatih (mungkin data tidak cukup).")