# Tugas Akhir 06 (TA-06) — Eksplorasi Mendalam Algoritma Random Forest

**Nama:** Muhammad Fajar Ramdani  
**NIM:** 301240036  
**Mata Praktikum:** Pembelajaran Mesin  
**Dosen Pengampu:** Mohammad Bayu Anggara, S.Kom., M.Kom.  
**Studi Kasus:** Prediksi Risiko Penyakit Jantung (*Heart Disease Classification*)  

## Tautan Proyek
- **Live Web Deployment (Vercel):** `[Tempel URL Vercel Anda di sini]`
- **Video Demo (3–5 Menit):** `[Tempel URL YouTube / Google Drive di sini]`

## Deskripsi Singkat
Proyek ini mengimplementasikan algoritma **Random Forest Classifier** untuk memprediksi risiko penyakit jantung berdasarkan 13 parameter klinis pasien. Eksperimen difokuskan pada optimasi hyperparameter dan analisis interpretabilitas model melalui:
1. **Skenario A (`n_estimators`):** Menguji pengaruh variasi jumlah pohon `[10, 50, 100, 200, 500]` terhadap stabilitas akurasi.
2. **Skenario B (`max_depth`):** Menguji variasi kedalaman pohon `[None, 3, 5, 10, 20]` untuk mengidentifikasi batas terjadinya *overfitting*.
3. **Feature Importance:** Mengekstrak fitur klinis paling berpengaruh untuk menghasilkan wawasan medis yang dapat ditindaklanjuti.

## Struktur Repositori
- `TA06_301240036_MuhammadFajarRamdani.ipynb` — Notebook Jupyter lengkap (Preprocessing, Skenario A & B, Feature Importance, dan Evaluasi Model).
- `heart_disease_dataset.csv` — Dataset klinis pasien (500 sampel, 14 kolom).
- `Laporan_TA06_301240036_MuhammadFajarRamdani.pdf` — Laporan analisis hasil eksperimen sesuai template jurnal.
- `web_deployment/` — Folder aplikasi web interaktif berbasis Python Flask (`app.py`, `rf_heart_model.pkl`, `model_columns.pkl`, `requirements.txt`, `vercel.json`).