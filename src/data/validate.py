# src/data/validate.py
"""
Data Quality Validation for IMS Bearing Features
Automated checks for nulls, outliers, drift, and anomalies

Usage:
    python src/data/validate.py
    python src/data/validate.py --output reports/data_quality_report.txt
"""
import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('DataValidator')

DATABASE_URL = os.getenv(
    'DATABASE_URL',
    'postgresql://postgres:postgres@localhost:5432/anudeep'
)


class DataQualityValidator:
    """Comprehensive data quality validation suite"""

    def __init__(self, db_url: str = DATABASE_URL):
        self.engine = create_engine(db_url, echo=False)
        self.report_lines = []
        self.issues = []

    def log(self, message: str, level: str = "INFO"):
        """Log message and add to report"""
        self.report_lines.append(message)
        if level == "INFO":
            logger.info(message)
        elif level == "WARNING":
            logger.warning(message)
            self.issues.append(message)
        elif level == "ERROR":
            logger.error(message)
            self.issues.append(message)

    def print_header(self, title: str):
        """Print section header"""
        line = "=" * 70
        self.log(line)
        self.log(f"  {title}")
        self.log(line)

    def load_data(self) -> pd.DataFrame:
        """Load features table from database"""
        self.log("\n📊 Loading data from PostgreSQL...")

        query = """
        SELECT * FROM features
        ORDER BY bearing_id, axis, timestamp
        """

        df = pd.read_sql(query, con=self.engine)
        self.log(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")

        return df

    def check_1_basic_stats(self, df: pd.DataFrame):
        """Check 1: Basic dataset statistics"""
        self.print_header("CHECK 1: BASIC DATASET STATISTICS")

        # Overall stats
        self.log(f"\nTotal rows:        {len(df):,}")
        self.log(f"Total columns:     {len(df.columns)}")
        self.log(f"Memory usage:      {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

        # Time range
        start_date = df['timestamp'].min()
        end_date = df['timestamp'].max()
        duration_days = (end_date - start_date).days
        self.log(f"\nTime range:")
        self.log(f"  Start:           {start_date}")
        self.log(f"  End:             {end_date}")
        self.log(f"  Duration:        {duration_days} days")

        # Bearing distribution
        self.log(f"\nBearing distribution:")
        bearing_counts = df.groupby(['bearing_id', 'axis']).size()
        for (bearing, axis), count in bearing_counts.items():
            self.log(f"  Bearing {bearing}-{axis}: {count:,} rows")

        # Failed vs Censored
        failed_count = df['failed'].sum()
        censored_count = df['censored'].sum()
        self.log(f"\nLabel distribution:")
        self.log(f"  Failed bearings:   {failed_count:,} rows ({100 * failed_count / len(df):.1f}%)")
        self.log(f"  Censored bearings: {censored_count:,} rows ({100 * censored_count / len(df):.1f}%)")

        # RUL stats
        rul_data = df[df['rul_hours'].notna()]['rul_hours']
        if len(rul_data) > 0:
            self.log(f"\nRUL statistics (failed bearings):")
            self.log(f"  Min RUL:         {rul_data.min():.2f} hours")
            self.log(f"  Max RUL:         {rul_data.max():.2f} hours")
            self.log(f"  Mean RUL:        {rul_data.mean():.2f} hours")
            self.log(f"  Median RUL:      {rul_data.median():.2f} hours")

    def check_2_null_analysis(self, df: pd.DataFrame):
        """Check 2: Null value analysis"""
        self.print_header("CHECK 2: NULL VALUE ANALYSIS")

        # Count nulls per column
        null_counts = df.isnull().sum()
        null_pcts = 100 * null_counts / len(df)

        # Columns with nulls
        cols_with_nulls = null_counts[null_counts > 0].sort_values(ascending=False)

        if len(cols_with_nulls) == 0:
            self.log("\n✓ NO NULL VALUES FOUND - Perfect data quality!")
        else:
            self.log(f"\n⚠ Found {len(cols_with_nulls)} columns with null values:\n")

            for col, count in cols_with_nulls.head(20).items():
                pct = null_pcts[col]
                self.log(f"  {col:40s}: {count:6,} nulls ({pct:5.2f}%)")

                # Flag if >5% nulls
                if pct > 5:
                    self.log(f"    ⚠ WARNING: >5% null rate!", "WARNING")

        # Check expected nulls (RUL for censored bearings)
        expected_nulls = df[df['censored'] == True]['rul_hours'].isnull().sum()
        total_censored = df['censored'].sum()

        self.log(f"\nExpected nulls (censored bearing RUL):")
        self.log(f"  Censored rows:   {total_censored:,}")
        self.log(f"  RUL nulls:       {expected_nulls:,}")

        if expected_nulls == total_censored:
            self.log("  ✓ All censored bearings correctly have NULL RUL")
        else:
            self.log(f"  ⚠ Mismatch: {total_censored - expected_nulls} censored rows have RUL", "WARNING")

    def check_3_outlier_detection(self, df: pd.DataFrame):
        """Check 3: Outlier detection using z-score"""
        self.print_header("CHECK 3: OUTLIER DETECTION (Z-SCORE > 3)")

        # Get numeric columns (exclude metadata)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [c for c in numeric_cols if c not in
                        ['feature_id', 'channel', 'bearing_id', 'n_windows']]

        self.log(f"\nAnalyzing {len(feature_cols)} numeric features...")

        outlier_summary = []

        for col in feature_cols[:50]:  # Check first 50 features (can adjust)
            if df[col].notna().sum() == 0:
                continue

            # Compute z-scores
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            outliers = (z_scores > 3).sum()
            outlier_pct = 100 * outliers / len(z_scores)

            if outliers > 0:
                outlier_summary.append({
                    'feature': col,
                    'outliers': outliers,
                    'pct': outlier_pct,
                    'max_zscore': z_scores.max()
                })

        if len(outlier_summary) == 0:
            self.log("\n✓ No outliers detected (all values within 3 std devs)")
        else:
            self.log(f"\n⚠ Found outliers in {len(outlier_summary)} features:\n")

            # Sort by outlier percentage
            outlier_summary.sort(key=lambda x: x['pct'], reverse=True)

            for item in outlier_summary[:10]:
                self.log(
                    f"  {item['feature']:40s}: {item['outliers']:4} outliers "
                    f"({item['pct']:5.2f}%), max z-score: {item['max_zscore']:.2f}"
                )

                # Flag if >1% outliers
                if item['pct'] > 1:
                    self.log(f"    ⚠ WARNING: >1% outlier rate - investigate!", "WARNING")

    def check_4_duplicate_timestamps(self, df: pd.DataFrame):
        """Check 4: Duplicate timestamp detection"""
        self.print_header("CHECK 4: DUPLICATE TIMESTAMP DETECTION")

        # Check for duplicates per bearing-axis
        duplicates = []

        for bearing in df['bearing_id'].unique():
            for axis in df['axis'].unique():
                subset = df[(df['bearing_id'] == bearing) & (df['axis'] == axis)]
                dup_count = subset['timestamp'].duplicated().sum()

                if dup_count > 0:
                    duplicates.append({
                        'bearing': bearing,
                        'axis': axis,
                        'duplicates': dup_count
                    })

        if len(duplicates) == 0:
            self.log("\n✓ No duplicate timestamps found")
        else:
            self.log(f"\n⚠ Found duplicate timestamps:\n", "WARNING")
            for dup in duplicates:
                self.log(
                    f"  Bearing {dup['bearing']}-{dup['axis']}: "
                    f"{dup['duplicates']} duplicate timestamps"
                )

    def check_5_feature_drift(self, df: pd.DataFrame):
        """Check 5: Feature drift analysis (early vs late data)"""
        self.print_header("CHECK 5: FEATURE DRIFT ANALYSIS (EARLY VS LATE)")

        # Split data into early (first 20%) and late (last 20%)
        df_sorted = df.sort_values('timestamp')
        split_idx_early = int(0.2 * len(df_sorted))
        split_idx_late = int(0.8 * len(df_sorted))

        early_data = df_sorted.iloc[:split_idx_early]
        late_data = df_sorted.iloc[split_idx_late:]

        self.log(f"\nComparing distributions:")
        self.log(f"  Early  {len(early_data):,} rows (first 20%)")
        self.log(f"  Late   {len(late_data):,} rows (last 20%)")

        # Check key features
        key_features = ['rms_mean', 'kurtosis_mean', 'crest_factor_mean',
                        'bp_1k_5k_mean', 'bp_5k_10k_mean']

        drift_detected = []

        self.log(f"\nFeature drift summary:\n")

        for feature in key_features:
            if feature not in df.columns:
                continue

            early_mean = early_data[feature].mean()
            late_mean = late_data[feature].mean()

            early_std = early_data[feature].std()
            late_std = late_data[feature].std()

            # Percent change
            pct_change = 100 * (late_mean - early_mean) / (early_mean + 1e-10)

            self.log(
                f"  {feature:25s}: "
                f"Early={early_mean:8.4f}, Late={late_mean:8.4f}, "
                f"Change={pct_change:+6.2f}%"
            )

            # Flag if >20% change
            if abs(pct_change) > 20:
                drift_detected.append(feature)
                self.log(f"    ⚠ Significant drift detected!", "WARNING")

        if len(drift_detected) == 0:
            self.log("\n✓ No significant drift detected (<20% change)")
        else:
            self.log(
                f"\n⚠ Drift detected in {len(drift_detected)} features: "
                f"{', '.join(drift_detected)}",
                "WARNING"
            )

    def check_6_feature_ranges(self, df: pd.DataFrame):
        """Check 6: Feature range validation"""
        self.print_header("CHECK 6: FEATURE RANGE VALIDATION")

        # Define expected ranges for key features
        expected_ranges = {
            'rms_mean': (0.01, 2.0),
            'kurtosis_mean': (-5, 50),
            'crest_factor_mean': (1.0, 10),
            'bp_0_1k_mean': (0, 0.1),
            'bp_1k_5k_mean': (0, 0.1),
            'bp_5k_10k_mean': (0, 0.1),
        }

        self.log("\nValidating feature ranges against expected bounds:\n")

        range_violations = []

        for feature, (min_val, max_val) in expected_ranges.items():
            if feature not in df.columns:
                continue

            actual_min = df[feature].min()
            actual_max = df[feature].max()

            out_of_range = (
                    (df[feature] < min_val) | (df[feature] > max_val)
            ).sum()

            status = "✓" if out_of_range == 0 else "⚠"

            self.log(
                f"  {status} {feature:25s}: "
                f"Range=[{actual_min:8.4f}, {actual_max:8.4f}], "
                f"Expected=[{min_val:6.2f}, {max_val:6.2f}]"
            )

            if out_of_range > 0:
                pct = 100 * out_of_range / len(df)
                self.log(f"    ⚠ {out_of_range} values ({pct:.2f}%) out of range", "WARNING")
                range_violations.append(feature)

        if len(range_violations) == 0:
            self.log("\n✓ All features within expected ranges")

    def check_7_temporal_consistency(self, df: pd.DataFrame):
        """Check 7: Temporal feature consistency"""
        self.print_header("CHECK 7: TEMPORAL FEATURE CONSISTENCY")

        # Check that temporal features are computed correctly
        self.log("\nValidating temporal feature computation:\n")

        # Check 1: Rolling means should be close to base features
        if 'rms_mean' in df.columns and 'rms_mean_roll_5_mean' in df.columns:
            diff = (df['rms_mean'] - df['rms_mean_roll_5_mean']).abs()
            mean_diff = diff.mean()
            max_diff = diff.max()

            self.log(f"  RMS vs Rolling Mean (window=5):")
            self.log(f"    Mean difference: {mean_diff:.6f}")
            self.log(f"    Max difference:  {max_diff:.6f}")

            if mean_diff > 0.1:  # Arbitrary threshold
                self.log(f"    ⚠ Large divergence detected", "WARNING")
            else:
                self.log(f"    ✓ Rolling mean tracking base feature correctly")

        # Check 2: Slopes should be reasonable (not too large)
        if 'rms_mean_slope_5' in df.columns:
            slopes = df['rms_mean_slope_5'].dropna()
            extreme_slopes = (slopes.abs() > 0.01).sum()

            self.log(f"\n  RMS Slope Analysis:")
            self.log(f"    Mean slope:      {slopes.mean():.6f}")
            self.log(f"    Std slope:       {slopes.std():.6f}")
            self.log(f"    Extreme slopes:  {extreme_slopes} ({100 * extreme_slopes / len(slopes):.2f}%)")

            if extreme_slopes / len(slopes) > 0.05:
                self.log(f"    ⚠ >5% extreme slopes - check computation", "WARNING")

    def generate_summary(self):
        """Generate final validation summary"""
        self.print_header("VALIDATION SUMMARY")

        if len(self.issues) == 0:
            self.log("\n🎉 ALL CHECKS PASSED - Data quality is excellent!")
            self.log("\n✓ No nulls (except expected censored RUL)")
            self.log("✓ No outliers detected")
            self.log("✓ No duplicate timestamps")
            self.log("✓ Feature ranges are valid")
            self.log("✓ Temporal features consistent")
        else:
            self.log(f"\n⚠ FOUND {len(self.issues)} ISSUES:\n")
            for i, issue in enumerate(self.issues, 1):
                self.log(f"  {i}. {issue}")

        self.log("\n" + "=" * 70)
        self.log(f"Validation completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("=" * 70)

    def run_all_checks(self, output_file: str = None):
        """Run all validation checks"""
        logger.info("=" * 70)
        logger.info("  IMS BEARING DATA QUALITY VALIDATION")
        logger.info("=" * 70)

        # Load data
        df = self.load_data()

        # Run all checks
        self.check_1_basic_stats(df)
        self.check_2_null_analysis(df)
        self.check_3_outlier_detection(df)
        self.check_4_duplicate_timestamps(df)
        self.check_5_feature_drift(df)
        self.check_6_feature_ranges(df)
        self.check_7_temporal_consistency(df)

        # Generate summary
        self.generate_summary()

        # Save report
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                f.write('\n'.join(self.report_lines))

            logger.info(f"\n✓ Report saved to: {output_path}")

        return len(self.issues) == 0


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="IMS Bearing Data Quality Validation"
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output file path for validation report (default: print to console)'
    )

    args = parser.parse_args()

    # Run validation
    validator = DataQualityValidator()
    success = validator.run_all_checks(output_file=args.output)

    # Exit code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
