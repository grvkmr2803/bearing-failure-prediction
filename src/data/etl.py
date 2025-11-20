# src/data/etl.py
"""
Production ETL Pipeline for IMS Bearing Data
- Parses raw ASCII files
- Extracts windowed features
- Generates temporal features
- Outputs to parquet (idempotent)

Usage:
    python src/data/etl.py --input data/raw/set1/1st_test --output data/processed/set1_features.parquet
    python src/data/etl.py --full-pipeline  # Run all stages
"""
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import list_files, load_file
from src.preprocess import window_signal, extract_features_from_window, aggregate_window_features
from src.temporal_features import (
    add_rolling_features, add_delta_features, add_ema_features,
    add_rolling_slope_features, add_zscore_features, add_bearing_aggregates
)
from src.config import SET1_CHANNEL_MAP, FS, WIN, OVERLAP
from src.utils import parse_ims_timestamp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ETL')


class IMSETLPipeline:
    """Production ETL pipeline for IMS bearing data"""

    def __init__(
            self,
            input_path: Path,
            output_path: Path,
            fs: int = FS,
            window_size: int = WIN,
            overlap: float = OVERLAP,
            force_reprocess: bool = False
    ):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.fs = fs
        self.window_size = window_size
        self.overlap = overlap
        self.force_reprocess = force_reprocess

        # Validate inputs
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input path not found: {self.input_path}")

        # Create output directory
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def check_if_processed(self) -> bool:
        """Check if output already exists (idempotency check)"""
        if self.output_path.exists() and not self.force_reprocess:
            logger.warning(
                f"Output file already exists: {self.output_path}\n"
                f"Use --force to reprocess."
            )
            return True
        return False

    def extract_base_features(self) -> pd.DataFrame:
        """
        Stage 1: Extract base features from raw files
        Returns: DataFrame with 20 base features per file-channel
        """
        logger.info("=" * 60)
        logger.info("STAGE 1: Extracting Base Features")
        logger.info("=" * 60)

        # Get all files
        files = list_files(str(self.input_path))
        logger.info(f"Found {len(files)} files to process")

        if len(files) == 0:
            raise ValueError(f"No files found in {self.input_path}")

        rows = []
        errors = []

        for file_path in tqdm(files, desc="Processing files"):
            try:
                # Load file
                data = load_file(file_path, dtype=np.float32)
                timestamp = parse_ims_timestamp(file_path)

                # Process each channel
                for ch in range(data.shape[1]):
                    sig = data[:, ch]

                    # Window signal
                    windows = window_signal(sig, self.window_size, self.overlap)

                    # Extract features per window
                    per_window_features = [
                        extract_features_from_window(w, fs=self.fs)
                        for w in windows
                    ]

                    # Aggregate across windows
                    agg_features = aggregate_window_features(per_window_features)

                    # Add metadata
                    row = {
                        "file": os.path.basename(file_path),
                        "timestamp": timestamp,
                        "channel": ch,
                        "bearing": SET1_CHANNEL_MAP[ch]["bearing"],
                        "axis": SET1_CHANNEL_MAP[ch]["axis"],
                        "n_windows": windows.shape[0],
                        **agg_features
                    }
                    rows.append(row)

            except Exception as e:
                error_msg = f"Failed to process {file_path}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
                continue

        # Create DataFrame
        df = pd.DataFrame(rows)
        df = df.sort_values(["timestamp", "bearing", "axis"]).reset_index(drop=True)

        logger.info(f"✓ Extracted features from {len(files)} files")
        logger.info(f"✓ Generated {len(df)} rows (files × channels)")
        logger.info(f"✓ Feature columns: {len(df.columns)}")

        if errors:
            logger.warning(f"⚠ {len(errors)} files failed to process")

        return df

    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Stage 2: Add temporal features (rolling, EMA, slopes, etc.)
        Returns: DataFrame with 220+ features
        """
        logger.info("=" * 60)
        logger.info("STAGE 2: Engineering Temporal Features")
        logger.info("=" * 60)

        # Get base feature columns (exclude metadata)
        feature_cols = [
            c for c in df.columns
            if c not in ['file', 'timestamp', 'channel', 'bearing', 'axis', 'n_windows']
        ]
        logger.info(f"Base features: {len(feature_cols)}")

        # Sort by bearing-axis-time (required for rolling features)
        df = df.sort_values(['bearing', 'axis', 'timestamp']).reset_index(drop=True)

        # Apply temporal feature engineering
        logger.info("Adding rolling features...")
        df = add_rolling_features(df, feature_cols, windows=[3, 5, 10])

        logger.info("Adding delta features...")
        df = add_delta_features(df, feature_cols)

        logger.info("Adding EMA features...")
        df = add_ema_features(df, feature_cols, alphas=[0.1, 0.3, 0.5])

        logger.info("Adding slope features...")
        df = add_rolling_slope_features(df, feature_cols, windows=[5, 10])

        logger.info("Adding z-score features...")
        df = add_zscore_features(df, feature_cols)

        logger.info("Adding bearing aggregates...")
        df = add_bearing_aggregates(df, feature_cols)

        logger.info(f"✓ Total features: {len(df.columns)}")
        logger.info(f"✓ Temporal features added: {len(df.columns) - len(feature_cols) - 6}")

        return df

    def save_output(self, df: pd.DataFrame, format: str = "parquet"):
        """Save processed features to disk"""
        logger.info("=" * 60)
        logger.info("STAGE 3: Saving Output")
        logger.info("=" * 60)

        output_path = self.output_path

        # Handle format
        if format == "parquet":
            try:
                df.to_parquet(output_path, index=False, engine='pyarrow')
                logger.info(f"✓ Saved parquet: {output_path}")
            except ImportError:
                logger.warning("pyarrow not installed, falling back to CSV")
                output_path = output_path.with_suffix('.csv')
                df.to_csv(output_path, index=False)
                logger.info(f"✓ Saved CSV: {output_path}")
        else:
            df.to_csv(output_path, index=False)
            logger.info(f"✓ Saved CSV: {output_path}")

        # Print summary
        file_size_mb = output_path.stat().st_size / (1024 ** 2)
        logger.info(f"✓ File size: {file_size_mb:.2f} MB")
        logger.info(f"✓ Rows: {len(df):,}")
        logger.info(f"✓ Columns: {len(df.columns)}")

    def run(self, include_temporal: bool = True) -> pd.DataFrame:
        """
        Run full ETL pipeline

        Args:
            include_temporal: If True, generate temporal features

        Returns:
            Processed DataFrame
        """
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("IMS BEARING ETL PIPELINE")
        logger.info("=" * 60)
        logger.info(f"Input: {self.input_path}")
        logger.info(f"Output: {self.output_path}")
        logger.info(f"Window size: {self.window_size}, Overlap: {self.overlap}")
        logger.info(f"Temporal features: {include_temporal}")
        logger.info("=" * 60)

        # Check if already processed
        if self.check_if_processed():
            logger.info("Loading existing output...")
            df = pd.read_parquet(self.output_path)
            logger.info(f"✓ Loaded {len(df)} rows from cache")
            return df

        # Stage 1: Base features
        df = self.extract_base_features()

        # Stage 2: Temporal features (optional)
        if include_temporal:
            df = self.add_temporal_features(df)

        # Stage 3: Save output
        self.save_output(df)

        # Summary
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info("=" * 60)
        logger.info(f"✓✓✓ ETL PIPELINE COMPLETE ✓✓✓")
        logger.info(f"✓ Elapsed time: {elapsed:.1f} seconds")
        logger.info(f"✓ Output: {self.output_path}")
        logger.info("=" * 60)

        return df


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="IMS Bearing ETL Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract base features only
  python src/data/etl.py --input data/raw/set1/1st_test --output data/processed/set1_features.parquet

  # Extract base + temporal features
  python src/data/etl.py --input data/raw/set1/1st_test --output data/processed/set1_features_temporal.parquet --temporal

  # Force reprocess existing output
  python src/data/etl.py --input data/raw/set1/1st_test --output data/processed/set1_features.parquet --force
        """
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input directory containing raw IMS files'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output parquet file path'
    )

    parser.add_argument(
        '--temporal', '-t',
        action='store_true',
        help='Generate temporal features (rolling, EMA, slopes)'
    )

    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Force reprocessing even if output exists'
    )

    parser.add_argument(
        '--window-size', '-w',
        type=int,
        default=WIN,
        help=f'Window size in samples (default: {WIN})'
    )

    parser.add_argument(
        '--overlap',
        type=float,
        default=OVERLAP,
        help=f'Window overlap fraction (default: {OVERLAP})'
    )

    parser.add_argument(
        '--fs',
        type=int,
        default=FS,
        help=f'Sampling frequency in Hz (default: {FS})'
    )

    args = parser.parse_args()

    # Run pipeline
    pipeline = IMSETLPipeline(
        input_path=args.input,
        output_path=args.output,
        fs=args.fs,
        window_size=args.window_size,
        overlap=args.overlap,
        force_reprocess=args.force
    )

    df = pipeline.run(include_temporal=args.temporal)

    # Print preview
    print("\n" + "=" * 60)
    print("PREVIEW (first 5 rows):")
    print("=" * 60)
    print(df.head())


if __name__ == "__main__":
    main()
