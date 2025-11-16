# Panduan Deployment ke Railway

## ✅ File Konfigurasi Railway

1. **railway.json** - Konfigurasi Railway dengan Nixpacks builder
2. **nixpacks.toml** - Konfigurasi build phases untuk Railway
3. **Procfile** - Start command untuk aplikasi
4. **runtime.txt** - Python version specification

## 📋 Langkah-langkah Deployment ke Railway

### 1. Pastikan File Sudah di Git

```bash
git add railway.json nixpacks.toml
git commit -m "Add Railway configuration"
git push origin main
```

### 2. Buat Project di Railway

1. Login ke [railway.app](https://railway.app)
2. Klik **"New Project"**
3. Pilih **"Deploy from GitHub repo"**
4. Pilih repository Anda: `attarmakhtar/ASIK25`

### 3. Konfigurasi di Railway Dashboard

Railway akan otomatis detect:
- **Build Command:** (dari railway.json atau nixpacks.toml)
- **Start Command:** `gunicorn app:app` (dari Procfile)

**Jika perlu set manual:**
- **Build Command:** `pip install --upgrade pip setuptools wheel && pip install -r requirements.txt`
- **Start Command:** `gunicorn app:app`

### 4. Environment Variables

Di Railway Dashboard → Variables, tambahkan:

```
SECRET_KEY = (generate random string)
FLASK_DEBUG = False
PORT = (Railway otomatis set, tidak perlu manual)
```

**Cara generate SECRET_KEY:**
```python
import secrets
print(secrets.token_hex(32))
```

### 5. Deploy

1. Railway akan otomatis:
   - Detect Python project
   - Install dependencies
   - Build aplikasi
   - Deploy aplikasi
2. Tunggu hingga status menjadi **"Active"**

## ⚠️ Troubleshooting

### Error: "ModuleNotFoundError: No module named 'encodings'"

**Solusi:**
1. Pastikan `runtime.txt` ada dengan format: `python-3.11.9`
2. Pastikan `railway.json` atau `nixpacks.toml` sudah ada
3. Di Railway Dashboard → Settings → Build:
   - Pastikan menggunakan **Nixpacks** builder
   - Atau set **Python Version** ke `3.11.9`

### Error: Python Installation Issues

**Solusi:**
1. Hapus service di Railway
2. Recreate service baru
3. Pastikan file konfigurasi sudah di-commit

### Build Timeout

**Solusi:**
- Prophet installation bisa memakan waktu lama
- Railway free tier memiliki build timeout
- Pertimbangkan upgrade ke paid tier jika build terlalu lama

## 🔧 Manual Fix di Railway Dashboard

Jika masih error, coba:

1. **Settings → Build:**
   - Builder: **Nixpacks**
   - Build Command: `pip install --upgrade pip setuptools wheel && pip install -r requirements.txt`

2. **Settings → Deploy:**
   - Start Command: `gunicorn app:app`
   - Healthcheck Path: (kosongkan atau `/`)

3. **Variables:**
   - Tambahkan `PYTHON_VERSION=3.11.9` (jika diperlukan)

## 📝 Checklist

- [ ] railway.json sudah di-commit
- [ ] nixpacks.toml sudah di-commit
- [ ] runtime.txt format benar: `python-3.11.9`
- [ ] requirements.txt lengkap
- [ ] Procfile ada
- [ ] SECRET_KEY sudah di-set di Railway
- [ ] FLASK_DEBUG=False

## 🎉 Setelah Deploy

Aplikasi akan tersedia di URL: `https://[project-name].up.railway.app`

Railway akan otomatis generate domain untuk Anda!

