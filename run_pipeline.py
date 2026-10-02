"""
Execution and verification script for Stage 1 High-Performance Ingestion Pipeline.
"""
import os
import sys
import time
import polars as pl
from src.preprocessing.polars_pipeline import HighDimensionalPreprocessor


def main():
    csv_path = os.path.join("data", "DataSet.csv")
    output_parquet = os.path.join("data", "processed_data.parquet")
    preprocessor_pkl = os.path.join("data", "preprocessor.pkl")

    # Optional max_features from CLI (e.g. 250 to achieve ~150-300 attributes), or default None
    max_features = int(sys.argv[1]) if len(sys.argv) > 1 else None

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input CSV not found at: {csv_path}")

    print("=" * 70)
    print("STAGE 1: HIGH-PERFORMANCE INGESTION PIPELINE (POLARS + PYARROW)")
    print("=" * 70)

    start_time = time.time()
    preprocessor = HighDimensionalPreprocessor(
        target_col="3924",
        variance_threshold=1e-4,
        max_features=max_features
    )

    # Fit and transform lazily
    df = preprocessor.fit_transform_lazy(csv_path=csv_path, output_parquet=output_parquet)

    # Save state
    preprocessor.save(preprocessor_pkl)
    elapsed_time = time.time() - start_time

    # Verification and summary metrics
    print("\n" + "=" * 70)
    print("PIPELINE VERIFICATION & HEALTH CHECKS")
    print("=" * 70)

    # Read output parquet with Polars
    pq_df = pl.read_parquet(output_parquet)
    print(f"[OK] Output Parquet File: {output_parquet}")
    print(f"[OK] Parquet File Size: {os.path.getsize(output_parquet) / (1024 * 1024):.2f} MB")
    print(f"[OK] Transformed Shape: {pq_df.shape[0]} rows x {pq_df.shape[1]} columns")
    print(f"[OK] Active Feature Count: {len(preprocessor.active_features)} (Filtered from 3,900+ attributes)")
    print(f"[OK] Total Pipeline Runtime: {elapsed_time:.2f} seconds")

    # Check for missing/null values
    null_counts = pq_df.null_count().sum_horizontal()[0]
    print(f"[OK] Null Values in Transformed Dataset: {null_counts}")
    assert null_counts == 0, f"Found {null_counts} null values in processed dataset!"

    # Target class distribution
    target_counts = pq_df[preprocessor.target_col].value_counts().sort(preprocessor.target_col)
    print("\n[OK] Target Distribution (0=Legitimate, 1=Mule):")
    for row in target_counts.iter_rows():
        label = "Legitimate" if row[0] == 0 else ("Mule" if row[0] == 1 else "Unknown")
        print(f"     Class {row[0]} ({label}): {row[1]} transactions ({row[1] / pq_df.shape[0] * 100:.2f}%)")

    # Test re-loading preprocessor state and transform test
    loaded_preprocessor = HighDimensionalPreprocessor.load(preprocessor_pkl)
    assert len(loaded_preprocessor.active_features) == len(preprocessor.active_features)
    print("[OK] Preprocessor state serialization & deserialization validated.")

    print("\n" + "=" * 70)
    print("STAGE 1 INGESTION PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
