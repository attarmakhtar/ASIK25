import os
import pandas as pd
import joblib
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ==============================================
# Path base directory
# ==============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
STATIC_DIR = os.path.join(BASE_DIR, '..', 'static')

# Lokasi alternatif untuk model
MODEL_PATHS = [
    os.path.join(DATA_DIR, 'rf_stok_ikan.pkl'),
    os.path.join(BASE_DIR, 'models', 'rf_stok_ikan.pkl'),
]

# ==============================================
# Load Data
# ==============================================
def load_data():
    stok = pd.read_csv(os.path.join(DATA_DIR, 'data_gabungan_new.csv'))
    stok.columns = stok.columns.str.lower()
    data_kecamatan = pd.read_csv(os.path.join(DATA_DIR, 'data_kecamatan_new.csv'))
    alat_ikan = pd.read_csv(os.path.join(DATA_DIR, 'jenis_alat_tangkap.csv'))
    return stok, data_kecamatan, alat_ikan


# ==============================================
# Load Model dengan fallback ke dua lokasi
# ==============================================
def load_model():
    model = None
    for path in MODEL_PATHS:
        if os.path.exists(path):
            model = joblib.load(path)
            print(f"[INFO] Model loaded from: {path}")
            break
    if model is None:
        raise FileNotFoundError("Model rf_stok_ikan.pkl not found in data/ or utils/models/")
    return model


# ==============================================
# Encode kolom kategorikal agar cocok dengan model
# ==============================================
def encode_features(df):
    label_encoders = {}
    categorical_cols = ['nama_ikan', 'jenis_ikan', 'lokasi', 'jenis_alat_tangkap']

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    return df, label_encoders


# ==============================================
# Buat prediksi stok ikan berdasarkan input user
# ==============================================
def predict_stok(model, data, nama_ikan, tahun, bulan, lokasi, jenis_alat_tangkap):
    df_input = pd.DataFrame([{
        'nama_ikan': nama_ikan,
        'tahun': tahun,
        'bulan': bulan,
        'lokasi': lokasi,
        'jenis_ikan': 'Unknown',
        'jenis_alat_tangkap': jenis_alat_tangkap
    }])

    df_input, _ = encode_features(df_input)

    pred = model.predict(df_input)
    return round(float(pred[0]), 2)


# ==============================================
# Buat data JSON untuk grafik (fitur 1)
# ==============================================
def generate_graph_json(stok, nama_ikan=None, tahun_pred=None, prediksi_nilai=None):
    # filter data historis
    if nama_ikan:
        stok = stok[stok['nama_ikan'].str.lower() == nama_ikan.lower()]

    grafik_data = stok.groupby('tahun')['total_kg_y'].sum().reset_index()
    grafik_data.rename(columns={'tahun': 'x', 'total_kg_y': 'y'}, inplace=True)
    grafik_data['label'] = 'Data Historis'

    # tambahkan data prediksi jika ada
    if tahun_pred and prediksi_nilai:
        grafik_data = pd.concat([
            grafik_data,
            pd.DataFrame({'x': [tahun_pred], 'y': [prediksi_nilai], 'label': ['Prediksi']})
        ], ignore_index=True)

    json_path = os.path.join(STATIC_DIR, 'grafik_stok_ikan.json')
    with open(json_path, 'w') as f:
        json.dump(grafik_data.to_dict(orient='records'), f, indent=4)

    print(f"[INFO] Grafik JSON saved to: {json_path}")
    return json_path
