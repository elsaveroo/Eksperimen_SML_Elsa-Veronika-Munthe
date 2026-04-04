# Eksperimen_SML_Elsa-Veronika-Munthe

Repository eksperimen Machine Learning untuk submission kelas **Membangun Sistem Machine Learning (MSML)**.

## 📊 Dataset: Water Quality (Potability)

- **Sumber**: [Kaggle – Water Potability](https://www.kaggle.com/datasets/adityakadiwal/water-potability)
- **Task**: Binary Classification – Prediksi kelayakan air minum
- **Fitur**: 9 fitur numerik (ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity)
- **Target**: `Potability` (0 = Tidak Layak, 1 = Layak Minum)

## 📁 Struktur Repository

```
Eksperimen_SML_Elsa-Veronika-Munthe/
├── .github/
│   └── workflows/
│       └── preprocessing.yml          # GitHub Actions CI workflow
├── water_potability_raw/
│   └── water_potability.csv           # Dataset mentah
├── preprocessing/
│   ├── Eksperimen_Elsa-Veronika-Munthe.ipynb        # Notebook eksperimen
│   ├── automate_Elsa-Veronika-Munthe.py             # Script otomatisasi preprocessing
│   └── water_potability_preprocessing/
│       ├── water_potability_preprocessing.csv   # Full dataset setelah preprocessing
│       ├── water_potability_train.csv           # Training set (80%)
│       └── water_potability_test.csv            # Testing set (20%)
└── README.md
```

## 🔧 Tahapan Preprocessing

1. **Median Imputation** – Menangani missing values pada kolom `ph`, `Sulfate`, `Trihalomethanes`
2. **IQR-based Outlier Clipping** – Mendeteksi dan meng-clip outlier pada semua fitur
3. **StandardScaler** – Standardisasi fitur (mean=0, std=1)
4. **Duplicate Removal** – Menghapus baris duplikat
5. **Stratified Train-Test Split** – Membagi data 80/20 dengan stratifikasi kelas

## 🚀 Cara Menjalankan

### Manual (lokal)
```bash
# Install dependencies
pip install pandas numpy scikit-learn

# Jalankan preprocessing
python preprocessing/automate_Elsa-Veronika-Munthe.py \
  --input water_potability_raw/water_potability.csv \
  --output_dir preprocessing/water_potability_preprocessing
```

### Otomatis (GitHub Actions)
Workflow akan berjalan otomatis ketika:
- Ada push ke branch `main` yang mengubah file dataset atau script preprocessing
- Dipicu secara manual melalui **Actions → Run workflow**

## 📈 Hasil EDA Singkat

| Insight | Detail |
|---------|--------|
| Ketidakseimbangan kelas | ~61% tidak layak, ~39% layak minum |
| Missing values | `ph` (491), `Sulfate` (781), `Trihalomethanes` (162) |
| Korelasi dengan target | Semua fitur memiliki korelasi rendah |
| Outlier | Terdeteksi pada hampir semua fitur, ditangani dengan IQR clipping |
