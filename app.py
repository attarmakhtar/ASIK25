from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import numpy as np
import os
import joblib
import pandas as pd
from prophet import Prophet
from sqlalchemy.sql import func, extract
import logging
import datetime # <-- BARU: Dibutuhkan untuk data tanggal

# --- BARU: Import untuk Database ---
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import extract
# -----------------------------------

# ===========================================
# KONFIGURASI DASAR
# ===========================================
app = Flask(__name__)

logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

# ===========================================
# PATH MODEL
# ===========================================
BASE_DIR = r"C:\Games\UPI Kampus Serang\Tugas\Semester 5\ASIK\WEBSITE"

# Prophet
PROPHET_MODEL_PATH = os.path.join(BASE_DIR, "prophet_models.pkl")
DATA_BERSIH = os.path.join(BASE_DIR, "data_bersih_model_ready.csv")

# Model Input
MODEL_INPUT_PATH = "model_stok_ikan_baru.pkl"
PREPROCESSOR_PATH = "preprocessor_stok_ikan_baru.pkl"

# --- BARU: KONFIGURASI DATABASE ---
app.config['SECRET_KEY'] = 'kunci-rahasia-anda-yang-sangat-aman-ganti-ini' 
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(BASE_DIR, 'ikan.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
# ------------------------------------

# --- BARU: MODEL DATABASE (3 TABEL) ---
# 1. Tabel Lama (dari CSV)
class ProduksiIkan(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    jenis_ikan = db.Column(db.String(100), nullable=False)
    nama_ikan = db.Column(db.String(100), nullable=False)
    tahun = db.Column(db.Integer, nullable=False)
    bulan = db.Column(db.Integer, nullable=False)
    total_kg = db.Column(db.Float, nullable=False)
    
    def __repr__(self):
        return f'<Produksi {self.nama_ikan} - {self.tahun}/{self.bulan}>'

# 2. Tabel Master Kapal
class DataKapal(db.Model):
    __tablename__ = 'data_kapal'
    
    id_kapal = db.Column(db.Integer, primary_key=True)
    nama_kapal = db.Column(db.String(150), nullable=False, unique=True)
    jenis_kapal = db.Column(db.String(100), nullable=False)
    ukuran_kapal_gt = db.Column(db.Integer, nullable=True)
    nomor_registrasi = db.Column(db.String(100), nullable=True)
    nama_pemilik = db.Column(db.String(150), nullable=True)
    pelabuhan_asal = db.Column(db.String(100), nullable=True)
    status = db.Column(db.String(50), nullable=False, default='Aktif')
    
    tangkapan = db.relationship('DataHarian', backref='kapal', lazy=True)

    def __repr__(self):
        return f'<Kapal {self.id_kapal}: {self.nama_kapal}>'

# 3. Tabel Transaksi Harian
class DataHarian(db.Model):
    __tablename__ = 'data_harian'
    
    id_tangkapan = db.Column(db.Integer, primary_key=True)
    id_kapal = db.Column(db.Integer, db.ForeignKey('data_kapal.id_kapal'), nullable=False)
    tanggal_tangkapan = db.Column(db.Date, nullable=False)
    jenis_ikan = db.Column(db.String(100), nullable=False)
    nama_ikan = db.Column(db.String(100), nullable=False)
    total_kg = db.Column(db.Float, nullable=False)
    lokasi_penangkapan = db.Column(db.String(150), nullable=True)
    durasi_trip_hari = db.Column(db.Integer, nullable=True)
    catatan = db.Column(db.Text, nullable=True)

    def __repr__(self):
        return f'<Tangkapan {self.id_tangkapan} - {self.nama_ikan} ({self.total_kg}kg)>'
# ------------------------------------

# ===========================================
# LOAD MODEL PREDIKSI INPUT
# ===========================================
# (Kode RF/OHE Anda tetap sama)
if not os.path.exists(MODEL_INPUT_PATH) or not os.path.exists(PREPROCESSOR_PATH):
    print(f"PERINGATAN: File model '{MODEL_INPUT_PATH}' atau preprocessor tidak ditemukan!")
    model_input = None
    preprocessor = None
else:
    model_input = joblib.load(MODEL_INPUT_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

INPUT_FEATURES = ['nama_ikan', 'lokasi', 'jenis_alat_tangkap', 'tahun']

# ===========================================
# LOAD MODEL PROPHET DAN DATA HISTORIS
# ===========================================
FISH_MAP = {}
HISTORICAL_DATA = {}
LATEST_HISTORICAL_YEAR = 2024

try:
    PROPHET_MODELS = joblib.load(PROPHET_MODEL_PATH)
except:
    print(f"PERINGATAN: Gagal memuat model Prophet dari '{PROPHET_MODEL_PATH}'.")
    PROPHET_MODELS = {}

def load_data_for_app():
    global FISH_MAP, HISTORICAL_DATA, LATEST_HISTORICAL_YEAR
    try:
        df = pd.read_csv(DATA_BERSIH, parse_dates=['ds'])
    except Exception as e:
        print(f"ERROR: Gagal memuat data bersih '{DATA_BERSIH}'. Error: {e}")
        return False

    # Load file asli
    try:
        raw_path = os.path.join(BASE_DIR, "data_produksi_perikanan_bersih.csv")
        df_raw = pd.read_csv(raw_path)
        df_raw.columns = df_raw.columns.str.lower().str.strip()
        FISH_MAP = df_raw.groupby('jenis_ikan')['nama_ikan'].unique().apply(list).to_dict()
    except Exception as e:
        print(f"PERINGATAN: Gagal memuat data mentah untuk FISH_MAP. Error: {e}")
        
    # Buat data historis tahunan
    df['tahun'] = df['ds'].dt.year
    df_yearly = df.groupby(['nama_ikan', 'tahun'])['y'].sum().reset_index()
    if not df_yearly.empty:
        LATEST_HISTORICAL_YEAR = int(df_yearly['tahun'].max())

    for ikan, group in df_yearly.groupby('nama_ikan'):
        HISTORICAL_DATA[ikan] = [
            {'x': int(row['tahun']), 'y': float(row['y'])}
            for idx, row in group.iterrows()
        ]

    return True

load_success = load_data_for_app()

# ===========================================
# ROUTES WEBSITE (DIMODIFIKASI)
# ===========================================

@app.route('/')
def home():
    # return render_template('beranda.html')
    # Untuk sementara, kita arahkan ke halaman manajemen
    return redirect(url_for('manajemen_kapal'))

# --- BARU: Route untuk link sidebar "Beranda" ---
@app.route('/beranda')
def beranda():
    # TODO: Buat dashboard di sini
    flash("Halaman Beranda (Dashboard) sedang dalam pengembangan.", "info")
    return render_template('beranda.html') # Asumsi Anda punya 'beranda.html'

@app.route('/grafik')
def grafik():
    return render_template('grafik.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/prediksi')
def prediksi_page():
    return render_template('prediksi.html')

# ===========================================
# --- ROUTES CRUD DATA HISTORIS (DIMODIFIKASI) ---
# ===========================================
@app.route('/crud')
def crud():
    # READ: Ambil semua data 'ProduksiIkan' (dari DB)
    try:
        data = ProduksiIkan.query.order_by(
            ProduksiIkan.tahun.desc(), 
            ProduksiIkan.bulan.desc(), 
            ProduksiIkan.nama_ikan
        ).all()
        
        jenis_query = db.session.query(ProduksiIkan.jenis_ikan).distinct().all()
        nama_query = db.session.query(ProduksiIkan.nama_ikan).distinct().all()
        
        jenis_ikan_list = sorted([j[0] for j in jenis_query])
        nama_ikan_list = sorted([n[0] for n in nama_query])
        
        return render_template('crud.html', 
                               data_list=data, 
                               jenis_ikan_list=jenis_ikan_list,
                               nama_ikan_list=nama_ikan_list)
    except Exception as e:
        flash(f"Gagal memuat data historis. Apakah 'ikan.db' sudah dimigrasi? Error: {e}", "danger")
        return render_template('crud.html', data_list=[], jenis_ikan_list=[], nama_ikan_list=[])

@app.route('/tambah', methods=['POST'])
def tambah_data():
    # CREATE: Tambah data 'ProduksiIkan'
    if request.method == 'POST':
        try:
            jenis_ikan = request.form['jenis_ikan']
            nama_ikan = request.form['nama_ikan']
            tahun = int(request.form['tahun'])
            bulan = int(request.form['bulan'])
            total_kg = float(request.form['total_kg'])

            existing = ProduksiIkan.query.filter_by(nama_ikan=nama_ikan, tahun=tahun, bulan=bulan).first()
            if existing:
                flash(f"Data untuk {nama_ikan} pada {bulan}/{tahun} sudah ada.", "warning")
                return redirect(url_for('crud'))

            data_baru = ProduksiIkan(jenis_ikan=jenis_ikan, nama_ikan=nama_ikan, tahun=tahun, bulan=bulan, total_kg=total_kg)
            db.session.add(data_baru)
            db.session.commit()
            flash("Data historis berhasil ditambahkan!", "success")
        except Exception as e:
            db.session.rollback()
            flash(f"Gagal menambahkan data: {e}", "danger")
        return redirect(url_for('crud'))

@app.route('/edit/<int:id>', methods=['POST'])
def edit_data(id):
    # UPDATE: Edit data 'ProduksiIkan'
    data = ProduksiIkan.query.get_or_404(id)
    if request.method == 'POST':
        try:
            data.jenis_ikan = request.form['jenis_ikan']
            data.nama_ikan = request.form['nama_ikan']
            data.tahun = int(request.form['tahun'])
            data.bulan = int(request.form['bulan'])
            data.total_kg = float(request.form['total_kg'])
            db.session.commit()
            flash("Data historis berhasil diperbarui!", "success")
        except Exception as e:
            db.session.rollback()
            flash(f"Gagal memperbarui data: {e}", "danger")
        return redirect(url_for('crud'))

@app.route('/hapus/<int:id>', methods=['POST'])
def hapus_data(id):
    # DELETE: Hapus data 'ProduksiIkan'
    data = ProduksiIkan.query.get_or_404(id)
    try:
        db.session.delete(data)
        db.session.commit()
        flash("Data historis berhasil dihapus.", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"Gagal menghapus data: {e}", "danger")
    return redirect(url_for('crud'))


# ===========================================
# --- DROPDOWN DATA UNTUK GRAFIK ---
# ===========================================
@app.route('/data_dropdown')
def data_dropdown():
    try:
        safe_map = {str(k): [str(vv) for vv in v] for k, v in FISH_MAP.items()}
        return jsonify({
            'status': 'success',
            'ikan_map': safe_map,
            'latest_year': int(LATEST_HISTORICAL_YEAR)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# ===========================================
# --- DATA GRAFIK (Prophet + Historis) ---
# ===========================================
@app.route('/data_grafik', methods=['POST'])
def data_grafik():
    try:
        nama_ikan = request.form.get('nama_ikan', '').strip()
        tahun_prediksi = int(request.form.get('tahun_prediksi', LATEST_HISTORICAL_YEAR))

        historis_data = HISTORICAL_DATA.get(nama_ikan, [])
        model_ikan = PROPHET_MODELS.get(nama_ikan)
        prediksi_list = []
        prediksi_val = 0
        message = ""

        if tahun_prediksi > LATEST_HISTORICAL_YEAR:
            if not model_ikan:
                prediksi_val = historis_data[-1]['y'] if historis_data else 0
                for y in range(LATEST_HISTORICAL_YEAR+1, tahun_prediksi+1):
                    prediksi_list.append({'x': y, 'y': prediksi_val})
                message = "Tidak ada model, memakai nilai tahun terakhir"
            else:
                message = "Prediksi Prophet"
                months = (tahun_prediksi - LATEST_HISTORICAL_YEAR) * 12
                future = model_ikan.make_future_dataframe(periods=months, freq='MS')
                forecast = model_ikan.predict(future)
                pred = forecast[forecast['ds'].dt.year > LATEST_HISTORICAL_YEAR]
                pred['tahun'] = pred['ds'].dt.year
                pred_year = pred.groupby('tahun')['yhat'].sum().reset_index()

                for idx, row in pred_year.iterrows():
                    prediksi_list.append({'x': int(row['tahun']), 'y': float(row['yhat'])})

                prediksi_val = prediksi_list[-1]['y'] if prediksi_list else 0
        else:
            h = [x for x in historis_data if x['x'] == tahun_prediksi]
            prediksi_val = h[0]['y'] if h else (historis_data[-1]['y'] if historis_data else 0)
            message = "Data historis"

        return jsonify({
            'status': 'success',
            'historis': historis_data,
            'prediksi_list': prediksi_list,
            'prediksi': {'x': tahun_prediksi, 'y': prediksi_val, 'message': message}
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# ===========================================
# --- PREDIKSI BERDASARKAN INPUT (MODEL RF) ---
# ===========================================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not model_input or not preprocessor:
             return jsonify({'status': 'error', 'message': 'Model RF tidak dimuat di server.'})
             
        lokasi = request.form.get('lokasi', '').strip()
        alat = request.form.get('alat_tangkap', '').strip()
        ikan = request.form.get('nama_ikan', '').strip()
        tahun = int(request.form.get('tahun', 2025))

        df_input = pd.DataFrame({
            'nama_ikan': [ikan],
            'lokasi': [lokasi],
            'jenis_alat_tangkap': [alat],
            'tahun': [tahun]
        }, columns=INPUT_FEATURES)

        processed = preprocessor.transform(df_input)
        pred = model_input.predict(processed)
        result = max(0, round(float(pred[0]), 2))

        return jsonify({'status': 'success', 'result': result})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# ===================================================
# 🚢 --- BARU: ROUTES CRUD UNTUK DATA KAPAL --- 🚢
# ===================================================
@app.route('/manajemen_kapal') # Rute untuk tab "Data Kapal"
def manajemen_kapal():
    # READ: Ambil semua data kapal dari DB
    try:
        data = DataKapal.query.order_by(DataKapal.nama_kapal).all()
        # Render template baru Anda (pastikan namanya manajemen_kapal.html)
        return render_template('manajemen_kapal.html', data_kapal_list=data)
    except Exception as e:
        flash(f"Gagal memuat data kapal: {e}", "danger")
        return render_template('manajemen_kapal.html', data_kapal_list=[])

@app.route('/kapal/tambah', methods=['POST'])
def tambah_kapal():
    # CREATE: Tambah data kapal baru
    if request.method == 'POST':
        try:
            nama_kapal = request.form['nama_kapal'].strip()
            jenis_kapal = request.form['jenis_kapal'].strip()
            ukuran_kapal_gt = request.form.get('ukuran_kapal_gt')
            nomor_registrasi = request.form.get('nomor_registrasi')
            nama_pemilik = request.form.get('nama_pemilik')
            pelabuhan_asal = request.form.get('pelabuhan_asal')
            status = request.form['status']

            existing = DataKapal.query.filter_by(nama_kapal=nama_kapal).first()
            if existing:
                flash(f"Nama kapal '{nama_kapal}' sudah terdaftar.", "warning")
                return redirect(url_for('manajemen_kapal'))

            kapal_baru = DataKapal(
                nama_kapal=nama_kapal,
                jenis_kapal=jenis_kapal,
                ukuran_kapal_gt=int(ukuran_kapal_gt) if ukuran_kapal_gt else None, 
                nomor_registrasi=nomor_registrasi if nomor_registrasi else None,
                nama_pemilik=nama_pemilik if nama_pemilik else None,
                pelabuhan_asal=pelabuhan_asal if pelabuhan_asal else None,
                status=status
            )
            db.session.add(kapal_baru)
            db.session.commit()
            flash(f"Kapal '{nama_kapal}' berhasil ditambahkan!", "success")
        except Exception as e:
            db.session.rollback()
            flash(f"Gagal menambahkan kapal: {e}", "danger")
        return redirect(url_for('manajemen_kapal'))

@app.route('/kapal/edit/<int:id_kapal>', methods=['POST'])
def edit_kapal(id_kapal):
    # UPDATE: Edit data kapal yang ada
    kapal = DataKapal.query.get_or_404(id_kapal)
    if request.method == 'POST':
        try:
            kapal.nama_kapal = request.form['nama_kapal'].strip()
            kapal.jenis_kapal = request.form['jenis_kapal'].strip()
            kapal.ukuran_kapal_gt = int(request.form.get('ukuran_kapal_gt')) if request.form.get('ukuran_kapal_gt') else None
            kapal.nomor_registrasi = request.form.get('nomor_registrasi') if request.form.get('nomor_registrasi') else None
            kapal.nama_pemilik = request.form.get('nama_pemilik') if request.form.get('nama_pemilik') else None
            kapal.pelabuhan_asal = request.form.get('pelabuhan_asal') if request.form.get('pelabuhan_asal') else None
            kapal.status = request.form['status']
            db.session.commit()
            flash(f"Data kapal '{kapal.nama_kapal}' berhasil diperbarui!", "success")
        except Exception as e:
            db.session.rollback()
            flash(f"Gagal memperbarui kapal: {e}", "danger")
        return redirect(url_for('manajemen_kapal'))

@app.route('/kapal/hapus/<int:id_kapal>', methods=['POST'])
def hapus_kapal(id_kapal):
    # DELETE: Hapus data kapal
    kapal = DataKapal.query.get_or_404(id_kapal)
    try:
        if kapal.tangkapan:
             flash(f"Gagal: Kapal '{kapal.nama_kapal}' tidak bisa dihapus karena memiliki {len(kapal.tangkapan)} catatan tangkapan.", "danger")
             return redirect(url_for('manajemen_kapal'))
        db.session.delete(kapal)
        db.session.commit()
        flash(f"Kapal '{kapal.nama_kapal}' berhasil dihapus.", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"Gagal menghapus kapal: {e}", "danger")
    return redirect(url_for('manajemen_kapal'))

# ===================================================
# 🗓️ --- BARU: ROUTES CRUD UNTUK DATA HARIAN --- 🗓️
# ===================================================

@app.route('/manajemen_harian')
def manajemen_harian():
    # READ: Tampilkan halaman manajemen data harian
    try:
        data = DataHarian.query.order_by(DataHarian.tanggal_tangkapan.desc()).all()
        kapal_list = DataKapal.query.filter_by(status='Aktif').order_by(DataKapal.nama_kapal).all()
        return render_template(
            'manajemen_harian.html', 
            data_harian_list=data, 
            kapal_list=kapal_list
        )
    except Exception as e:
        flash(f"Gagal memuat data harian: {e}", "danger")
        return render_template('manajemen_harian.html', data_harian_list=[], kapal_list=[])

@app.route('/harian/tambah', methods=['POST'])
def tambah_harian():
    # CREATE: Tambah data tangkapan harian baru
    if request.method == 'POST':
        try:
            id_kapal = int(request.form['id_kapal'])
            tanggal_tangkapan = datetime.datetime.strptime(request.form['tanggal_tangkapan'], '%Y-%m-%d').date()
            nama_ikan = request.form['nama_ikan'].strip()
            jenis_ikan = request.form['jenis_ikan'].strip()
            total_kg = float(request.form['total_kg'])
            lokasi_penangkapan = request.form.get('lokasi_penangkapan')
            durasi_trip_hari = request.form.get('durasi_trip_hari')
            catatan = request.form.get('catatan')

            data_baru = DataHarian(
                id_kapal=id_kapal,
                tanggal_tangkapan=tanggal_tangkapan,
                nama_ikan=nama_ikan,
                jenis_ikan=jenis_ikan,
                total_kg=total_kg,
                lokasi_penangkapan=lokasi_penangkapan if lokasi_penangkapan else None,
                durasi_trip_hari=int(durasi_trip_hari) if durasi_trip_hari else None,
                catatan=catatan if catatan else None
            )
            db.session.add(data_baru)
            db.session.commit()
            flash("Catatan tangkapan harian berhasil ditambahkan!", "success")
        except Exception as e:
            db.session.rollback()
            flash(f"Gagal menambahkan data harian: {e}", "danger")
        return redirect(url_for('manajemen_harian'))

@app.route('/harian/edit/<int:id_tangkapan>', methods=['POST'])
def edit_harian(id_tangkapan):
    # UPDATE: Edit data harian yang ada
    data = DataHarian.query.get_or_404(id_tangkapan)
    if request.method == 'POST':
        try:
            data.id_kapal = int(request.form['id_kapal'])
            data.tanggal_tangkapan = datetime.datetime.strptime(request.form['tanggal_tangkapan'], '%Y-%m-%d').date()
            data.nama_ikan = request.form['nama_ikan'].strip()
            data.jenis_ikan = request.form['jenis_ikan'].strip()
            data.total_kg = float(request.form['total_kg'])
            data.lokasi_penangkapan = request.form.get('lokasi_penangkapan') if request.form.get('lokasi_penangkapan') else None
            data.durasi_trip_hari = int(request.form.get('durasi_trip_hari')) if request.form.get('durasi_trip_hari') else None
            data.catatan = request.form.get('catatan') if request.form.get('catatan') else None
            
            db.session.commit()
            flash(f"Data tangkapan #{data.id_tangkapan} berhasil diperbarui!", "success")
        except Exception as e:
            db.session.rollback()
            flash(f"Gagal memperbarui data harian: {e}", "danger")
        return redirect(url_for('manajemen_harian'))

@app.route('/harian/hapus/<int:id_tangkapan>', methods=['POST'])
def hapus_harian(id_tangkapan):
    # DELETE: Hapus data harian
    data = DataHarian.query.get_or_404(id_tangkapan)
    try:
        db.session.delete(data)
        db.session.commit()
        flash(f"Data tangkapan #{id_tangkapan} berhasil dihapus.", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"Gagal menghapus data harian: {e}", "danger")
    return redirect(url_for('manajemen_harian'))

# ========================================================
# 📊 --- DIUBAH: ROUTES LAPORAN DENGAN FILTER --- 📊
# ========================================================

@app.route('/laporan_bulanan')
def laporan_bulanan():
    month_map = {
        1: 'Januari', 2: 'Februari', 3: 'Maret', 4: 'April', 
        5: 'Mei', 6: 'Juni', 7: 'Juli', 8: 'Agustus', 
        9: 'September', 10: 'Oktober', 11: 'November', 12: 'Desember'
    }

    selected_year = request.args.get('filter_tahun', type=int)
    selected_month = request.args.get('filter_bulan', type=int)

    # FIX: Menggunakan func.extract() untuk mendapatkan tahun yang tersedia
    available_years = db.session.query(
        func.extract('year', DataHarian.tanggal_tangkapan).label('tahun')
    ).distinct().order_by('tahun').all()
    available_years = [y[0] for y in available_years]

    # FIX: Menggunakan func.extract() untuk kolom tahun dan bulan di SELECT
    query = db.session.query(
        func.extract('year', DataHarian.tanggal_tangkapan).label('tahun'),
        func.extract('month', DataHarian.tanggal_tangkapan).label('bulan'),
        DataHarian.nama_ikan,
        DataHarian.jenis_ikan,
        func.sum(DataHarian.total_kg).label('total_bulanan')
    )

    if selected_year:
        query = query.filter(extract('year', DataHarian.tanggal_tangkapan) == selected_year)
    if selected_month:
        query = query.filter(extract('month', DataHarian.tanggal_tangkapan) == selected_month)

    # Menggunakan label 'tahun' dan 'bulan' yang sudah didefinisikan di SELECT
    query = query.group_by(
        'tahun', 
        'bulan', 
        DataHarian.nama_ikan, 
        DataHarian.jenis_ikan
    ).order_by('tahun').order_by('bulan')

    laporan_data = query.all()

    return render_template('laporan_bulanan.html', 
                           laporan_data=laporan_data, 
                           available_years=available_years, 
                           selected_year=selected_year,
                           selected_month=selected_month,
                           month_map=month_map)
# ... (Definisi Model dan app setup lainnya) ...

@app.route('/laporan_tahunan')
def laporan_tahunan():
    month_map = {
        1: 'Januari', 2: 'Februari', 3: 'Maret', 4: 'April', 
        5: 'Mei', 6: 'Juni', 7: 'Juli', 8: 'Agustus', 
        9: 'September', 10: 'Oktober', 11: 'November', 12: 'Desember'
    }

    selected_year = request.args.get('filter_tahun', type=int)

    # FIX: Menggunakan func.extract() untuk mendapatkan tahun yang tersedia
    available_years = db.session.query(
        func.extract('year', DataHarian.tanggal_tangkapan).label('tahun')
    ).distinct().order_by('tahun').all()
    available_years = [y[0] for y in available_years]

    # Base Query (data dikelompokkan per bulan sesuai permintaan user sebelumnya)
    # FIX: Menggunakan func.extract() untuk kolom tahun dan bulan di SELECT
    query = db.session.query(
        func.extract('year', DataHarian.tanggal_tangkapan).label('tahun'),
        func.extract('month', DataHarian.tanggal_tangkapan).label('bulan'),
        DataHarian.nama_ikan,
        DataHarian.jenis_ikan,
        func.sum(DataHarian.total_kg).label('total_bulanan')
    )

    # Aplikasikan Filter Tahun
    if selected_year:
        query = query.filter(extract('year', DataHarian.tanggal_tangkapan) == selected_year)

    # Grouping dan Ordering
    # Menggunakan label 'tahun' dan 'bulan'
    query = query.group_by(
        'tahun',
        'bulan',
        DataHarian.nama_ikan,
        DataHarian.jenis_ikan
    ).order_by('tahun').order_by('bulan').order_by(DataHarian.jenis_ikan)

    laporan_data = query.all()

    return render_template('laporan_tahunan.html', 
                           laporan_data=laporan_data, 
                           available_years=available_years, 
                           selected_year=selected_year,
                           month_map=month_map)
# ===========================================
# RUN SERVER (DIMODIFIKASI)
# ===========================================
if __name__ == '__main__':
    # --- BARU: Buat tabel database saat aplikasi pertama kali dijalankan ---
    with app.app_context():
        print("Mengecek dan membuat tabel database (jika diperlukan)...")
        db.create_all()
        print("Tabel 'ProduksiIkan', 'DataKapal', dan 'DataHarian' sudah siap.")
    # -----------------------------------------------------------------
    
    print("\nServer Flask siap dijalankan...")
    app.run(debug=True)