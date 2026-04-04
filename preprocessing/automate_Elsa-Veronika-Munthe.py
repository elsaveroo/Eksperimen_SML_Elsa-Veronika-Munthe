"""
automate_Elsa-Veronika-Munthe.py
==================
Script otomatisasi preprocessing dataset Water Quality (Potability).
Mengkonversi langkah-langkah eksperimen pada notebook menjadi fungsi
yang dapat dijalankan secara otomatis dan mengembalikan data siap latih.

"""

import argparse
import os
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# ─── Setup logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── 1. Load Data ─────────────────────────────────────────────────────────────
def load_data(filepath: str) -> pd.DataFrame:
    """Memuat dataset dari file CSV."""
    logger.info(f"Memuat dataset dari: {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Dataset berhasil dimuat: {df.shape[0]} baris x {df.shape[1]} kolom")
    logger.info(f"Kolom: {list(df.columns)}")
    return df


# ─── 2. Handle Missing Values ─────────────────────────────────────────────────
def handle_missing_values(df: pd.DataFrame, target_col: str = "Potability") -> pd.DataFrame:
    """
    Menangani missing values menggunakan Median Imputation.
    Median digunakan karena lebih robust terhadap outlier dibandingkan mean.
    """
    feature_cols = [col for col in df.columns if col != target_col]
    missing_before = df[feature_cols].isnull().sum()

    cols_with_missing = missing_before[missing_before > 0].index.tolist()
    if cols_with_missing:
        logger.info(f"Kolom dengan missing values: {cols_with_missing}")
        imputer = SimpleImputer(strategy="median")
        df[feature_cols] = imputer.fit_transform(df[feature_cols])
        logger.info("Median Imputation selesai.")
    else:
        logger.info("Tidak ada missing values yang ditemukan.")

    assert df[feature_cols].isnull().sum().sum() == 0, "Masih ada missing values setelah imputasi!"
    return df


# ─── 3. Handle Outliers ────────────────────────────────────────────────────────
def handle_outliers(
    df: pd.DataFrame,
    target_col: str = "Potability",
    multiplier: float = 1.5,
) -> pd.DataFrame:
    """
    Menangani outlier menggunakan IQR Method dengan Clipping.
    Nilai di luar [Q1 - 1.5*IQR, Q3 + 1.5*IQR] akan di-clip ke batas tersebut.
    """
    feature_cols = [col for col in df.columns if col != target_col]
    total_clipped = 0

    for col in feature_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR

        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        if n_outliers > 0:
            df[col] = df[col].clip(lower=lower, upper=upper)
            total_clipped += n_outliers
            logger.info(f"  {col}: {n_outliers} outlier di-clip ke [{lower:.3f}, {upper:.3f}]")

    logger.info(f"Total outlier yang ditangani: {total_clipped}")
    return df


# ─── 4. Standardisasi Fitur ────────────────────────────────────────────────────
def standardize_features(
    df: pd.DataFrame, target_col: str = "Potability"
) -> pd.DataFrame:
    """
    Melakukan standardisasi fitur menggunakan StandardScaler.
    Menghasilkan distribusi dengan mean=0 dan std=1.
    """
    feature_cols = [col for col in df.columns if col != target_col]
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    logger.info(f"Standardisasi selesai untuk {len(feature_cols)} fitur.")
    return df


# ─── 5. Remove Duplicates ─────────────────────────────────────────────────────
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Menghapus baris duplikat dari dataset."""
    n_before = len(df)
    df = df.drop_duplicates()
    n_removed = n_before - len(df)
    if n_removed > 0:
        logger.info(f"Menghapus {n_removed} baris duplikat.")
    else:
        logger.info("Tidak ada baris duplikat.")
    return df


# ─── 6. Train-Test Split ──────────────────────────────────────────────────────
def split_data(
    df: pd.DataFrame,
    target_col: str = "Potability",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Membagi dataset menjadi training dan testing set.
    Menggunakan stratified split untuk mempertahankan proporsi kelas.
    """
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(f"Train size: {X_train.shape[0]} ({(1-test_size)*100:.0f}%)")
    logger.info(f"Test size : {X_test.shape[0]} ({test_size*100:.0f}%)")
    logger.info(f"Distribusi kelas train: {y_train.value_counts().to_dict()}")
    logger.info(f"Distribusi kelas test : {y_test.value_counts().to_dict()}")

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    return train_df, test_df


# ─── 7. Save Results ──────────────────────────────────────────────────────────
def save_results(
    df_full: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """Menyimpan hasil preprocessing ke folder output."""
    os.makedirs(output_dir, exist_ok=True)

    full_path = os.path.join(output_dir, "water_potability_preprocessing.csv")
    train_path = os.path.join(output_dir, "water_potability_train.csv")
    test_path = os.path.join(output_dir, "water_potability_test.csv")

    df_full.to_csv(full_path, index=False)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f"Dataset lengkap disimpan  : {full_path}")
    logger.info(f"Dataset training disimpan : {train_path}")
    logger.info(f"Dataset testing disimpan  : {test_path}")


# ─── Main Pipeline ────────────────────────────────────────────────────────────
def run_preprocessing(
    input_path: str,
    output_dir: str,
    target_col: str = "Potability",
    test_size: float = 0.2,
    random_state: int = 42,
    iqr_multiplier: float = 1.5,
) -> tuple:
    """
    Menjalankan full preprocessing pipeline dan mengembalikan data siap latih.

    Returns:
        train_df (pd.DataFrame): Data training yang sudah diproses.
        test_df  (pd.DataFrame): Data testing yang sudah diproses.
    """
    logger.info("=" * 60)
    logger.info("     MEMULAI PIPELINE PREPROCESSING WATER QUALITY")
    logger.info("=" * 60)

    # Step 1: Load
    df = load_data(input_path)

    # Step 2: Handle missing values
    df = handle_missing_values(df, target_col)

    # Step 3: Remove duplicates
    df = remove_duplicates(df)

    # Step 4: Handle outliers
    df = handle_outliers(df, target_col, multiplier=iqr_multiplier)

    # Step 5: Standardize
    df = standardize_features(df, target_col)

    # Step 6: Split
    train_df, test_df = split_data(df, target_col, test_size, random_state)

    # Step 7: Save
    save_results(df, train_df, test_df, output_dir)

    logger.info("=" * 60)
    logger.info("     PREPROCESSING SELESAI!")
    logger.info(f"     Output disimpan di: {output_dir}")
    logger.info("=" * 60)

    return train_df, test_df


# ─── CLI Entry Point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Otomatisasi Preprocessing Dataset Water Quality"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="water_potability.csv",
        help="Path ke file dataset raw (default: water_potability.csv)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="water_potability_preprocessing",
        help="Folder output hasil preprocessing (default: water_potability_preprocessing)",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proporsi data testing (default: 0.2)",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--iqr_multiplier",
        type=float,
        default=1.5,
        help="Multiplier IQR untuk deteksi outlier (default: 1.5)",
    )

    args = parser.parse_args()

    train_df, test_df = run_preprocessing(
        input_path=args.input,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        iqr_multiplier=args.iqr_multiplier,
    )

    print(f"\nData training siap: {train_df.shape}")
    print(f"Data testing siap : {test_df.shape}")
