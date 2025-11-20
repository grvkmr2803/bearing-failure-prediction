# src/data/split.py
"""
Train/Test Split for Time-Series Bearing Data
Uses time-based split to prevent data leakage

Usage:
    python src/data/split.py --split-date 2003-11-15
    python src/data/split.py --split-pct 0.8
    python src/data/split.py --bearing-stratified
"""
import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('TrainTestSplit')

DATABASE_URL = os.getenv(
    'DATABASE_URL',
    'postgresql://postgres:postgres@localhost:5432/anudeep'
)


class TimeSeriesSplitter:
    """Time-based train/test splitter for bearing failure data"""

    def __init__(self, db_url: str = DATABASE_URL):
        self.engine = create_engine(db_url, echo=False)
        self.split_info = {}

    def add_split_column(self):
        """Add split column to features table if not exists"""
        logger.info("Checking for split column...")

        with self.engine.begin() as conn:
            # Check if column exists
            result = conn.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'features' AND column_name = 'split'
            """))

            if result.fetchone() is None:
                logger.info("Adding 'split' column to features table...")
                conn.execute(text("ALTER TABLE features ADD COLUMN split VARCHAR(10)"))
                logger.info("✓ Column added")
            else:
                logger.info("✓ Column already exists")

    def get_time_range(self):
        """Get min and max timestamps from dataset"""
        with self.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT 
                    MIN(timestamp) as min_ts,
                    MAX(timestamp) as max_ts
                FROM features
            """))
            row = result.fetchone()
            return row[0], row[1]

    def split_by_date(self, split_date: str):
        """
        Split data by specific date

        Args:
            split_date: Date string in format 'YYYY-MM-DD'
        """
        logger.info("=" * 70)
        logger.info("  TIME-BASED TRAIN/TEST SPLIT")
        logger.info("=" * 70)
        logger.info(f"Split date: {split_date}")

        # Get time range
        min_ts, max_ts = self.get_time_range()
        logger.info(f"Data range: {min_ts} to {max_ts}")

        # Update split column
        with self.engine.begin() as conn:
            # Set train split
            train_result = conn.execute(text(f"""
                UPDATE features
                SET split = 'train'
                WHERE timestamp < '{split_date}'::timestamptz
            """))

            # Set test split
            test_result = conn.execute(text(f"""
                UPDATE features
                SET split = 'test'
                WHERE timestamp >= '{split_date}'::timestamptz
            """))

            logger.info(f"✓ Train rows: {train_result.rowcount:,}")
            logger.info(f"✓ Test rows:  {test_result.rowcount:,}")

        # Verify split
        self.verify_split()

    def split_by_percentage(self, train_pct: float = 0.8, stratify_by_bearing: bool = False):
        """
        Split data by percentage of time

        Args:
            train_pct: Percentage of data for training (0-1)
            stratify_by_bearing: If True, split each bearing separately
        """
        logger.info("=" * 70)
        logger.info("  TIME-BASED TRAIN/TEST SPLIT (PERCENTAGE)")
        logger.info("=" * 70)
        logger.info(f"Train percentage: {train_pct * 100:.1f}%")
        logger.info(f"Stratify by bearing: {stratify_by_bearing}")

        if stratify_by_bearing:
            self._split_stratified(train_pct)
        else:
            self._split_global(train_pct)

        # Verify split
        self.verify_split()

    def _split_global(self, train_pct: float):
        """Global split across all data"""
        # Load timestamps
        df = pd.read_sql("SELECT feature_id, timestamp FROM features ORDER BY timestamp", con=self.engine)

        # Compute split index
        split_idx = int(train_pct * len(df))
        split_timestamp = df.iloc[split_idx]['timestamp']

        logger.info(f"Split timestamp: {split_timestamp}")
        logger.info(f"Train: first {split_idx:,} rows (first {train_pct * 100:.1f}% by time)")
        logger.info(f"Test:  last {len(df) - split_idx:,} rows (last {(1 - train_pct) * 100:.1f}% by time)")

        # Update database
        with self.engine.begin() as conn:
            train_result = conn.execute(text(f"""
                UPDATE features
                SET split = 'train'
                WHERE timestamp < '{split_timestamp}'::timestamptz
            """))

            test_result = conn.execute(text(f"""
                UPDATE features
                SET split = 'test'
                WHERE timestamp >= '{split_timestamp}'::timestamptz
            """))

            logger.info(f"✓ Train rows: {train_result.rowcount:,}")
            logger.info(f"✓ Test rows:  {test_result.rowcount:,}")

    def _split_stratified(self, train_pct: float):
        """Split each bearing-axis separately"""
        logger.info("\nSplitting each bearing-axis separately...")

        # Load data grouped by bearing
        df = pd.read_sql("""
            SELECT feature_id, timestamp, bearing_id, axis
            FROM features
            ORDER BY bearing_id, axis, timestamp
        """, con=self.engine)

        split_assignments = []

        for (bearing, axis), group in df.groupby(['bearing_id', 'axis']):
            # Compute split for this group
            split_idx = int(train_pct * len(group))

            train_ids = group.iloc[:split_idx]['feature_id'].tolist()
            test_ids = group.iloc[split_idx:]['feature_id'].tolist()

            split_assignments.extend([('train', fid) for fid in train_ids])
            split_assignments.extend([('test', fid) for fid in test_ids])

            logger.info(f"  Bearing {bearing}-{axis}: {len(train_ids)} train, {len(test_ids)} test")

        # Bulk update
        logger.info("\nUpdating database...")
        with self.engine.begin() as conn:
            for split_val, feature_id in split_assignments:
                conn.execute(text(f"""
                    UPDATE features
                    SET split = '{split_val}'
                    WHERE feature_id = {feature_id}
                """))

        logger.info("✓ Stratified split complete")

    def verify_split(self):
        """Verify split quality and print statistics"""
        logger.info("\n" + "=" * 70)
        logger.info("  SPLIT VERIFICATION")
        logger.info("=" * 70)

        with self.engine.connect() as conn:
            # Overall split
            split_counts = pd.read_sql("""
                SELECT 
                    split,
                    COUNT(*) as count,
                    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2) as pct
                FROM features
                WHERE split IS NOT NULL
                GROUP BY split
                ORDER BY split
            """, con=conn)

            logger.info("\nOverall split:")
            for _, row in split_counts.iterrows():
                logger.info(f"  {row['split']:5s}: {row['count']:6,} rows ({row['pct']:5.2f}%)")

            # Split by failed/censored
            split_by_label = pd.read_sql("""
                SELECT 
                    split,
                    failed,
                    censored,
                    COUNT(*) as count
                FROM features
                WHERE split IS NOT NULL
                GROUP BY split, failed, censored
                ORDER BY split, failed DESC
            """, con=conn)

            logger.info("\nSplit by label:")
            for _, row in split_by_label.iterrows():
                label = "Failed" if row['failed'] else ("Censored" if row['censored'] else "Unknown")
                logger.info(f"  {row['split']:5s} | {label:8s}: {row['count']:6,} rows")

            # Split by bearing
            split_by_bearing = pd.read_sql("""
                SELECT 
                    bearing_id,
                    axis,
                    split,
                    COUNT(*) as count
                FROM features
                WHERE split IS NOT NULL
                GROUP BY bearing_id, axis, split
                ORDER BY bearing_id, axis, split
            """, con=conn)

            logger.info("\nSplit by bearing-axis:")
            for bearing in sorted(split_by_bearing['bearing_id'].unique()):
                for axis in ['x', 'y']:
                    subset = split_by_bearing[
                        (split_by_bearing['bearing_id'] == bearing) &
                        (split_by_bearing['axis'] == axis)
                        ]
                    train_count = subset[subset['split'] == 'train']['count'].sum()
                    test_count = subset[subset['split'] == 'test']['count'].sum()
                    total = train_count + test_count
                    train_pct = 100.0 * train_count / total if total > 0 else 0

                    logger.info(
                        f"  Bearing {bearing}-{axis}: "
                        f"Train={train_count:4} ({train_pct:5.1f}%), "
                        f"Test={test_count:4}"
                    )

            # Time ranges
            time_ranges = pd.read_sql("""
                SELECT 
                    split,
                    MIN(timestamp) as min_ts,
                    MAX(timestamp) as max_ts
                FROM features
                WHERE split IS NOT NULL
                GROUP BY split
                ORDER BY split
            """, con=conn)

            logger.info("\nTime ranges:")
            for _, row in time_ranges.iterrows():
                duration = (row['max_ts'] - row['min_ts']).days
                logger.info(
                    f"  {row['split']:5s}: {row['min_ts']} to {row['max_ts']} "
                    f"({duration} days)"
                )

            # Check for leakage
            train_max = time_ranges[time_ranges['split'] == 'train']['max_ts'].iloc[0]
            test_min = time_ranges[time_ranges['split'] == 'test']['min_ts'].iloc[0]

            if train_max < test_min:
                logger.info(f"\n✓ NO DATA LEAKAGE: Train ends before test begins")
                logger.info(f"  Train max: {train_max}")
                logger.info(f"  Test min:  {test_min}")
            else:
                logger.warning(f"\n⚠ POTENTIAL LEAKAGE: Train and test overlap!")
                logger.warning(f"  Train max: {train_max}")
                logger.warning(f"  Test min:  {test_min}")

        logger.info("\n" + "=" * 70)
        logger.info("✓✓✓ SPLIT VERIFICATION COMPLETE")
        logger.info("=" * 70)

    def create_split_table(self):
        """Create data_splits table for reproducibility"""
        logger.info("\nCreating data_splits tracking table...")

        with self.engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS data_splits (
                    split_id SERIAL PRIMARY KEY,
                    split_name VARCHAR(50),
                    split_method VARCHAR(50),
                    split_date TIMESTAMPTZ,
                    train_count INTEGER,
                    test_count INTEGER,
                    train_pct REAL,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    notes TEXT
                )
            """))

            # Get split counts
            result = conn.execute(text("""
                SELECT split, COUNT(*) as count
                FROM features
                WHERE split IS NOT NULL
                GROUP BY split
            """))

            counts = {row[0]: row[1] for row in result}
            train_count = counts.get('train', 0)
            test_count = counts.get('test', 0)
            total = train_count + test_count
            train_pct = train_count / total if total > 0 else 0

            # Insert split record
            conn.execute(text("""
                INSERT INTO data_splits (
                    split_name, split_method, train_count, test_count, train_pct, notes
                ) VALUES (
                    'set1_split_v1', 'time_based', :train, :test, :pct, 'Initial train/test split'
                )
            """), {"train": train_count, "test": test_count, "pct": train_pct})

            logger.info("✓ Split metadata saved to data_splits table")


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Train/Test Split for IMS Bearing Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Split by date (train = before date, test = after date)
  python src/data/split.py --split-date 2003-11-15

  # Split by percentage (80/20 split)
  python src/data/split.py --split-pct 0.8

  # Stratified split (each bearing split separately)
  python src/data/split.py --split-pct 0.8 --bearing-stratified
        """
    )

    parser.add_argument(
        '--split-date',
        type=str,
        help='Split date in YYYY-MM-DD format (data before = train, after = test)'
    )

    parser.add_argument(
        '--split-pct',
        type=float,
        default=0.8,
        help='Train percentage (0-1, default: 0.8)'
    )

    parser.add_argument(
        '--bearing-stratified',
        action='store_true',
        help='Split each bearing-axis separately (maintains bearing proportions)'
    )

    args = parser.parse_args()

    # Create splitter
    splitter = TimeSeriesSplitter()

    # Add split column
    splitter.add_split_column()

    # Perform split
    if args.split_date:
        splitter.split_by_date(args.split_date)
    else:
        splitter.split_by_percentage(
            train_pct=args.split_pct,
            stratify_by_bearing=args.bearing_stratified
        )

    # Create tracking table
    splitter.create_split_table()

    logger.info("\n✓✓✓ TRAIN/TEST SPLIT COMPLETE ✓✓✓")


if __name__ == "__main__":
    main()
