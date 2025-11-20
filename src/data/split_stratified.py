# src/data/split_stratified.py
"""
Stratified Train/Test Split by RUL for Bearing Failure Prediction
Ensures both train and test have examples across all RUL ranges

Usage:
    python src/data/split_stratified.py --train-pct 0.8
"""
import os
import sys
import argparse
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('StratifiedSplitter')

DATABASE_URL = os.getenv(
    'DATABASE_URL',
    'postgresql://postgres:postgres@localhost:5432/anudeep'
)


class StratifiedRULSplitter:
    """Stratified split by RUL bins for RUL prediction"""

    def __init__(self, db_url: str = DATABASE_URL):
        self.engine = create_engine(db_url, echo=False)

    def create_rul_bins(self, df: pd.DataFrame):
        """Create RUL bins for stratification"""
        logger.info("Creating RUL bins for stratification...")

        # Define RUL bins (in hours)
        bins = [0, 50, 100, 150, 200, 300, 400, 600, 850]
        labels = ['0-50h', '50-100h', '100-150h', '150-200h',
                  '200-300h', '300-400h', '400-600h', '600+h']

        df['rul_bin'] = pd.cut(df['rul_hours'], bins=bins, labels=labels, include_lowest=True)

        # Show distribution
        logger.info("\nRUL bin distribution:")
        bin_counts = df['rul_bin'].value_counts().sort_index()
        for bin_label, count in bin_counts.items():
            pct = 100 * count / len(df)
            logger.info(f"  {bin_label:12s}: {count:5} samples ({pct:5.2f}%)")

        return df

    def stratified_split(self, train_pct: float = 0.8):
        """Perform stratified split by RUL bins"""
        logger.info("=" * 70)
        logger.info("  STRATIFIED TRAIN/TEST SPLIT BY RUL")
        logger.info("=" * 70)
        logger.info(f"Train percentage: {train_pct * 100:.1f}%")

        # Load failed bearing data (only these have RUL values)
        logger.info("\nLoading failed bearing data...")
        df = pd.read_sql("""
            SELECT 
                feature_id,
                bearing_id,
                axis,
                timestamp,
                rul_hours,
                failed
            FROM features
            WHERE failed = TRUE AND rul_hours IS NOT NULL
            ORDER BY bearing_id, axis, timestamp
        """, con=self.engine)

        logger.info(f"✓ Loaded {len(df):,} samples from failed bearings")

        # Create RUL bins
        df = self.create_rul_bins(df)

        # Stratified split
        logger.info(f"\nPerforming stratified {train_pct * 100:.0f}/{(1 - train_pct) * 100:.0f} split...")

        train_ids, test_ids = train_test_split(
            df['feature_id'],
            test_size=(1 - train_pct),
            stratify=df['rul_bin'],
            random_state=42
        )

        logger.info(f"✓ Train: {len(train_ids):,} samples")
        logger.info(f"✓ Test:  {len(test_ids):,} samples")

        # Update database
        logger.info("\nUpdating database...")

        with self.engine.begin() as conn:
            # Reset all to NULL first
            conn.execute(text("UPDATE features SET split = NULL"))

            # Set train split for failed bearings
            train_ids_str = ','.join(map(str, train_ids.tolist()))
            conn.execute(text(f"""
                UPDATE features 
                SET split = 'train'
                WHERE feature_id IN ({train_ids_str})
            """))

            # Set test split for failed bearings
            test_ids_str = ','.join(map(str, test_ids.tolist()))
            conn.execute(text(f"""
                UPDATE features 
                SET split = 'test'
                WHERE feature_id IN ({test_ids_str})
            """))

            # Set censored bearings to train (they don't contribute to RUL prediction)
            conn.execute(text("""
                UPDATE features 
                SET split = 'train'
                WHERE failed = FALSE AND split IS NULL
            """))

            logger.info("✓ Database updated")

        # Verify split
        self.verify_split(train_ids, test_ids, df)

        return train_ids, test_ids

    def verify_split(self, train_ids, test_ids, df):
        """Verify split quality"""
        logger.info("\n" + "=" * 70)
        logger.info("  SPLIT VERIFICATION")
        logger.info("=" * 70)

        train_df = df[df['feature_id'].isin(train_ids)]
        test_df = df[df['feature_id'].isin(test_ids)]

        # Overall counts
        logger.info(f"\nOverall:")
        logger.info(f"  Train: {len(train_df):,} samples ({100 * len(train_df) / len(df):.1f}%)")
        logger.info(f"  Test:  {len(test_df):,} samples ({100 * len(test_df) / len(df):.1f}%)")

        # RUL distribution per split
        logger.info(f"\nRUL statistics:")
        logger.info(f"  Train: min={train_df['rul_hours'].min():.1f}, "
                    f"max={train_df['rul_hours'].max():.1f}, "
                    f"mean={train_df['rul_hours'].mean():.1f}")
        logger.info(f"  Test:  min={test_df['rul_hours'].min():.1f}, "
                    f"max={test_df['rul_hours'].max():.1f}, "
                    f"mean={test_df['rul_hours'].mean():.1f}")

        # Distribution per RUL bin
        logger.info(f"\nRUL bin distribution:")
        logger.info(f"{'Bin':12s} {'Train':>8s} {'Test':>8s} {'Train%':>8s} {'Test%':>8s}")
        logger.info("-" * 50)

        for bin_label in sorted(df['rul_bin'].unique()):
            train_count = (train_df['rul_bin'] == bin_label).sum()
            test_count = (test_df['rul_bin'] == bin_label).sum()
            total = train_count + test_count

            train_pct = 100 * train_count / total if total > 0 else 0
            test_pct = 100 * test_count / total if total > 0 else 0

            logger.info(f"{bin_label:12s} {train_count:8} {test_count:8} "
                        f"{train_pct:7.1f}% {test_pct:7.1f}%")

        # Per bearing distribution
        logger.info(f"\nPer bearing:")
        for bearing in sorted(df['bearing_id'].unique()):
            for axis in ['x', 'y']:
                train_count = ((train_df['bearing_id'] == bearing) &
                               (train_df['axis'] == axis)).sum()
                test_count = ((test_df['bearing_id'] == bearing) &
                              (test_df['axis'] == axis)).sum()
                total = train_count + test_count
                train_pct = 100 * train_count / total if total > 0 else 0

                logger.info(f"  Bearing {bearing}-{axis}: "
                            f"Train={train_count:4} ({train_pct:5.1f}%), "
                            f"Test={test_count:4}")

        # Check overlap (should be good now)
        train_rul_range = (train_df['rul_hours'].min(), train_df['rul_hours'].max())
        test_rul_range = (test_df['rul_hours'].min(), test_df['rul_hours'].max())

        overlap = (max(train_rul_range[0], test_rul_range[0]),
                   min(train_rul_range[1], test_rul_range[1]))

        if overlap[1] > overlap[0]:
            overlap_size = overlap[1] - overlap[0]
            logger.info(f"\n✓ RUL OVERLAP EXISTS: {overlap_size:.1f} hours")
            logger.info(f"  Train range: [{train_rul_range[0]:.1f}, {train_rul_range[1]:.1f}]")
            logger.info(f"  Test range:  [{test_rul_range[0]:.1f}, {test_rul_range[1]:.1f}]")
            logger.info(f"  Overlap:     [{overlap[0]:.1f}, {overlap[1]:.1f}]")
        else:
            logger.warning(f"\n⚠ NO RUL OVERLAP - This might still be a problem!")

        logger.info("\n" + "=" * 70)
        logger.info("✓✓✓ VERIFICATION COMPLETE")
        logger.info("=" * 70)


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Stratified train/test split by RUL for bearing failure prediction"
    )

    parser.add_argument(
        '--train-pct',
        type=float,
        default=0.8,
        help='Training set percentage (default: 0.8)'
    )

    args = parser.parse_args()

    # Run stratified split
    splitter = StratifiedRULSplitter()
    train_ids, test_ids = splitter.stratified_split(train_pct=args.train_pct)

    logger.info("\n✓✓✓ STRATIFIED SPLIT COMPLETE ✓✓✓")


if __name__ == "__main__":
    main()
