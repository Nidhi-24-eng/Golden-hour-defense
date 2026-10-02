import os
import pickle
from typing import List, Dict, Optional, Union
import numpy as np
import polars as pl
from sklearn.preprocessing import RobustScaler


class HighDimensionalPreprocessor:
    """
    High-Performance Ingestion and Preprocessing Engine using Polars LazyFrames and PyArrow.
    
    Processes high-dimensional tabular datasets (3,900+ dense columns) with:
      1. Zero-copy lazy scanning via pl.scan_csv()
      2. Zero/low-variance filtering to eliminate constant/uninformative columns
      3. Optional max_features rank-ordering by variance down to ~150-300 informative attributes
      4. Robust median computation and imputation against banking transaction outliers
      5. Target vector isolation (0=Legitimate, 1=Mule)
      6. RobustScaler transformation on numerical behavioral features
      7. Parquet serialization for high throughput downstream ML
    """

    def __init__(
        self,
        target_col: str = "3924",
        variance_threshold: float = 1e-4,
        max_features: Optional[int] = None
    ):
        self.target_col = target_col
        self.variance_threshold = variance_threshold
        self.max_features = max_features
        self.active_features: List[str] = []
        self.medians: Dict[str, float] = {}
        self.scaler: RobustScaler = RobustScaler()
        self.actual_target_col: Optional[str] = None
        self.feature_variances: Dict[str, float] = {}

    def _resolve_target_col(self, col_names: List[str]) -> str:
        """
        Dynamically detects target column name format (e.g. '3924' vs 'F3924').
        """
        if self.target_col in col_names:
            return self.target_col
        prefixed = f"F{self.target_col}"
        if prefixed in col_names:
            return prefixed
        raise ValueError(
            f"Target column '{self.target_col}' (or '{prefixed}') not found in dataset columns."
        )

    def fit_transform_lazy(self, csv_path: str, output_parquet: str) -> pl.DataFrame:
        """
        Asynchronously scans high-dimensional dataset, applies variance filtering,
        computes robust medians, and saves clean tabular data as Parquet.
        
        Parameters:
            csv_path: Path to the raw high-dimensional CSV file.
            output_parquet: Destination path for the processed Parquet file.
            
        Returns:
            pl.DataFrame: Cleaned, imputed, and scaled dataframe including target column.
        """
        print(f"[*] Scanning {csv_path} with Polars LazyFrame...")
        lazy_df = pl.scan_csv(
            csv_path,
            infer_schema_length=10000,
            null_values=["NA", "null", "NULL", ""]
        )

        schema = lazy_df.collect_schema()
        all_col_names = schema.names()
        self.actual_target_col = self._resolve_target_col(all_col_names)
        print(f"[+] Target column resolved to: '{self.actual_target_col}'")

        # Exclude empty index column (e.g., '""' or 'Unnamed: 0') and target column
        ignore_cols = {self.actual_target_col, self.target_col, "", "Unnamed: 0"}
        candidate_cols = [
            c for c in all_col_names
            if c not in ignore_cols and schema[c].is_numeric()
        ]
        print(f"[*] Identified {len(candidate_cols)} candidate numeric attributes for evaluation.")

        # 1. Variance Filtering
        print(f"[*] Computing column variances across {len(candidate_cols)} attributes...")
        var_exprs = [pl.col(c).var().alias(c) for c in candidate_cols]
        variances = lazy_df.select(var_exprs).collect().to_dict(as_series=False)

        valid_vars = {
            c: variances[c][0]
            for c in candidate_cols
            if variances[c][0] is not None and variances[c][0] > self.variance_threshold
        }
        self.feature_variances = valid_vars

        if self.max_features is not None and len(valid_vars) > self.max_features:
            # Rank features by variance descending to extract top informative attributes
            sorted_features = sorted(valid_vars.keys(), key=lambda c: valid_vars[c], reverse=True)
            self.active_features = sorted_features[:self.max_features]
            print(
                f"[+] Reduced features from {len(candidate_cols)} to {len(self.active_features)} "
                f"top informative columns (max_features={self.max_features}, variance > {self.variance_threshold})."
            )
        else:
            self.active_features = list(valid_vars.keys())
            print(
                f"[+] Reduced features from {len(candidate_cols)} to "
                f"{len(self.active_features)} active columns (variance > {self.variance_threshold})."
            )

        # 2. Median Computation for Robust Imputation
        print("[*] Computing feature medians for robust imputation...")
        median_exprs = [pl.col(c).median().alias(c) for c in self.active_features]
        raw_medians = lazy_df.select(median_exprs).collect().to_dict(as_series=False)
        self.medians = {
            k: float(v[0]) if v[0] is not None else 0.0
            for k, v in raw_medians.items()
        }

        # 3. Impute & Materialize
        print("[*] Materializing cleaned active features and target vector...")
        impute_exprs = [
            pl.col(c).fill_null(self.medians[c]).alias(c)
            for c in self.active_features
        ]

        # Ensure target column is properly isolated as Int32 (0=Legitimate, 1=Mule)
        clean_df = (
            lazy_df
            .select(impute_exprs + [pl.col(self.actual_target_col).cast(pl.Int32).alias(self.target_col)])
            .collect()
        )

        # 4. Scale numerical behavioral vectors using RobustScaler
        print("[*] Applying RobustScaler to mitigate banking transaction outliers...")
        X = clean_df.select(self.active_features).to_numpy()
        y = clean_df.select(self.target_col).to_numpy().ravel()

        X_scaled = self.scaler.fit_transform(X)

        # 5. Build final Polars DataFrame and write to Parquet
        output_dir = os.path.dirname(output_parquet)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        clean_pl = pl.DataFrame(X_scaled, schema=self.active_features).with_columns(
            pl.Series(self.target_col, y, dtype=pl.Int32)
        )
        
        clean_pl.write_parquet(output_parquet)
        print(f"[+] Clean Parquet dataset saved to {output_parquet}")
        print(f"[+] Output shape: {clean_pl.shape[0]} rows x {clean_pl.shape[1]} columns")

        return clean_pl

    def transform(self, df: Union[pl.DataFrame, pl.LazyFrame]) -> pl.DataFrame:
        """
        Transforms new incoming transaction batches using fitted medians and scaler.
        """
        if not self.active_features or self.scaler is None:
            raise RuntimeError("Preprocessor has not been fitted yet.")

        if isinstance(df, pl.LazyFrame):
            df = df.collect()

        impute_exprs = [
            pl.col(c).fill_null(self.medians.get(c, 0.0)).alias(c)
            for c in self.active_features
        ]
        transformed_df = df.select(impute_exprs)
        X = transformed_df.to_numpy()
        X_scaled = self.scaler.transform(X)

        result_df = pl.DataFrame(X_scaled, schema=self.active_features)

        # Include target if present
        target = self.actual_target_col or self.target_col
        if target in df.columns:
            y = df.select(pl.col(target).cast(pl.Int32)).to_series()
            result_df = result_df.with_columns(y.alias(self.target_col))

        return result_df

    def save(self, filepath: str) -> None:
        """
        Persists the fitted preprocessor state using pickle.
        """
        state = {
            "target_col": self.target_col,
            "variance_threshold": self.variance_threshold,
            "max_features": self.max_features,
            "active_features": self.active_features,
            "medians": self.medians,
            "scaler": self.scaler,
            "actual_target_col": self.actual_target_col,
            "feature_variances": self.feature_variances,
        }
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(state, f)
        print(f"[+] Preprocessor state saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "HighDimensionalPreprocessor":
        """
        Loads a pre-fitted preprocessor state from disk.
        """
        with open(filepath, "rb") as f:
            state = pickle.load(f)
        preprocessor = cls(
            target_col=state["target_col"],
            variance_threshold=state["variance_threshold"],
            max_features=state.get("max_features"),
        )
        preprocessor.active_features = state["active_features"]
        preprocessor.medians = state["medians"]
        preprocessor.scaler = state["scaler"]
        preprocessor.actual_target_col = state.get("actual_target_col")
        preprocessor.feature_variances = state.get("feature_variances", {})
        print(f"[+] Preprocessor state loaded from {filepath} ({len(preprocessor.active_features)} active features)")
        return preprocessor


if __name__ == "__main__":
    import sys

    raw_csv = sys.argv[1] if len(sys.argv) > 1 else "data/DataSet.csv"
    out_parquet = sys.argv[2] if len(sys.argv) > 2 else "data/processed_data.parquet"
    out_pkl = sys.argv[3] if len(sys.argv) > 3 else "data/preprocessor.pkl"

    preprocessor = HighDimensionalPreprocessor(target_col="3924", variance_threshold=1e-4)
    preprocessor.fit_transform_lazy(raw_csv, out_parquet)
    preprocessor.save(out_pkl)
