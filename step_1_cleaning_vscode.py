import pandas as pd
import numpy as np
import os

# --- KONFIGURASI ---
# (Kita gunakan path yang sama dengan script Step 2)
BASE_DIR = r"C:\Games\UPI Kampus Serang\Tugas\Semester 5\ASIK\WEBSITE"
INPUT_FILE_PATH = os.path.join(BASE_DIR, "Data_Produksi_Perikanan_Bersih.csv")
OUTPUT_FILE_PATH = os.path.join(BASE_DIR, "data_bersih_model_ready.csv")

print(f"Memulai Step 1: Pembersihan Data...")
print(f"Membaca file dari: {INPUT_FILE_PATH}")

# --- 1. MEMUAT DATA ---
try:
    df = pd.read_csv(INPUT_FILE_PATH)
except FileNotFoundError:
    print(f"ERROR: File mentah tidak ditemukan di {INPUT_FILE_PATH}")
    print("Pastikan file 'Data_Produksi_Perikanan_Bersih.csv' ada di folder WEBSITE Anda.")
    raise SystemExit
except Exception as e:
    print(f"Terjadi error saat membaca file: {e}")
    raise SystemExit

print(f"Data awal dimuat: {len(df)} baris.")

# --- 2. PRA-PEMROSESAN ---
df.columns = df.columns.str.strip().str.lower()
try:
    df['tanggal'] = pd.to_datetime(df['tanggal'])
except Exception as e:
    print(f"ERROR: Gagal memproses kolom 'tanggal'. Error: {e}")
    raise SystemExit

df['total_kg'] = pd.to_numeric(df['total_kg'], errors='coerce')
df.dropna(subset=['tanggal', 'total_kg'], inplace=True)
df['nama_ikan'] = df['nama_ikan'].str.strip()
print(f"Data setelah pra-pemrosesan: {len(df)} baris.")

# --- 3. DETEKSI & PENGHAPUSAN ANOMALI (METODE IQR) ---
print("Memulai deteksi anomali (IQR) per ikan...")

def remove_outliers_iqr(df_group):
    Q1 = df_group['total_kg'].quantile(0.25)
    Q3 = df_group['total_kg'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)
    df_cleaned = df_group[(df_group['total_kg'] >= lower_bound) & (df_group['total_kg'] <= upper_bound)]
    return df_cleaned

df_no_anomalies = df.groupby('nama_ikan').apply(remove_outliers_iqr)
df_no_anomalies = df_no_anomalies.reset_index(drop=True)

removed_count = len(df) - len(df_no_anomalies)
print(f"Deteksi anomali selesai. Total baris data anomali dihapus: {removed_count}")

# --- 4. PERSIAPAN FINAL UNTUK MODEL PROPHET ---
print("Menyiapkan data final untuk model Prophet...")
df_final = df_no_anomalies[['nama_ikan', 'tanggal', 'total_kg']].copy()
df_final.rename(columns={'tanggal': 'ds', 'total_kg': 'y'}, inplace=True)
df_final = df_final.sort_values(by=['nama_ikan', 'ds'])
print(f"Data final siap: {len(df_final)} baris.")

# --- 5. SIMPAN KE FILE BARU ---
try:
    df_final.to_csv(OUTPUT_FILE_PATH, index=False)
    print(f"\nBERHASIL!")
    print(f"Data bersih siap untuk model telah disimpan di:")
    print(f"{OUTPUT_FILE_PATH}")
    print("\nSekarang Anda bisa menjalankan Step 2.")
except Exception as e:
    print(f"\nERROR: Gagal menyimpan file di {OUTPUT_FILE_PATH}")
    print(f"Detail: {e}")