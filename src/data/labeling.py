# src/data/labeling.py
"""
Labeling script for IMS features table.
Computes rul_seconds, rul_hours, failed, censored per the notebook logic.

Assumptions:
- 'features' table has columns: feature_id, file_name, timestamp, bearing_id
- timestamp is timestamptz
- failed bearings for set1 are {3,4}; others are censored
- Uses DATABASE_URL env var or falls back to postgres@localhost/anudeep
"""
import os
import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("labeling")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/anudeep"
)


def load_features(engine):
    """Load feature IDs, timestamps, and bearing IDs"""
    logger.info("Loading features table from DB")
    df = pd.read_sql(
        "SELECT feature_id, file_name, timestamp, bearing_id FROM features",
        con=engine
    )
    logger.info("Loaded %d rows", len(df))
    return df


def compute_rul(df, failed_bearings={3, 4}):
    """
    Compute RUL and flags for each bearing.
    Returns DataFrame with feature_id + label columns only.
    """
    logger.info("Computing RUL and flags")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Find last timestamp per bearing (failure time)
    last_ts_per_bearing = df.groupby("bearing_id")["timestamp"].max().to_dict()

    # Initialize label columns
    df["rul_seconds"] = None
    df["rul_hours"] = None
    df["failed"] = False
    df["censored"] = False

    all_bearings = set(df["bearing_id"].unique())
    non_failed = sorted(list(all_bearings - set(failed_bearings)))

    # For failed bearings: rul = t_fail - t_now
    for b in failed_bearings:
        if b not in last_ts_per_bearing:
            logger.warning("Bearing %s not present in dataset; skipping", b)
            continue

        t_fail = last_ts_per_bearing[b]
        mask = df["bearing_id"] == b
        delta = t_fail - df.loc[mask, "timestamp"]
        df.loc[mask, "rul_seconds"] = delta.dt.total_seconds()
        df.loc[mask, "rul_hours"] = df.loc[mask, "rul_seconds"] / 3600.0
        df.loc[mask, "failed"] = True

        logger.info(
            f"Bearing {b}: {mask.sum()} rows labeled as failed, "
            f"RUL range: {df.loc[mask, 'rul_hours'].min():.1f}h to "
            f"{df.loc[mask, 'rul_hours'].max():.1f}h"
        )

    # For non-failed bearings: mark censored
    if non_failed:
        censored_mask = df["bearing_id"].isin(non_failed)
        df.loc[censored_mask, "censored"] = True
        logger.info(f"Bearings {non_failed}: {censored_mask.sum()} rows marked censored")

    # Return only columns needed for update
    return df[["feature_id", "rul_seconds", "rul_hours", "failed", "censored"]]


def apply_updates_batch(engine, updates_df, batch_size=1000):
    """
    Apply updates using batch UPDATE statements.
    More reliable than temp table for this use case.
    """
    logger.info(f"Applying {len(updates_df)} updates in batches of {batch_size}")

    # Build parameterized UPDATE statement
    update_sql = """
        UPDATE features
        SET 
            rul_seconds = :rul_seconds,
            rul_hours = :rul_hours,
            failed = :failed,
            censored = :censored
        WHERE feature_id = :feature_id
    """

    # Convert to list of dicts for executemany
    records = updates_df.to_dict('records')

    with engine.begin() as conn:
        # Execute in batches to avoid memory issues
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            conn.execute(text(update_sql), batch)
            logger.info(f"Updated batch {i // batch_size + 1}/{(len(records) - 1) // batch_size + 1}")

    logger.info("✓ DB updates applied successfully")


def apply_updates_staging(engine, updates_df, staging_table="features_labels_staging"):
    """
    Alternative: Use staging table approach (faster for large datasets).
    Creates a regular (non-temp) staging table, bulk inserts, updates, then drops.
    """
    logger.info(f"Applying updates via staging table: {staging_table}")

    with engine.begin() as conn:
        # Drop staging table if exists
        conn.execute(text(f"DROP TABLE IF EXISTS {staging_table}"))

        # Create staging table
        conn.execute(text(f"""
            CREATE TABLE {staging_table} (
                feature_id BIGINT PRIMARY KEY,
                rul_seconds DOUBLE PRECISION,
                rul_hours DOUBLE PRECISION,
                failed BOOLEAN,
                censored BOOLEAN
            )
        """))

    # Bulk insert via pandas (outside transaction to avoid conflicts)
    updates_df.to_sql(
        staging_table,
        con=engine,
        if_exists='append',
        index=False,
        method='multi',
        chunksize=5000
    )
    logger.info(f"Loaded {len(updates_df)} rows into staging table")

    # Update main table from staging
    with engine.begin() as conn:
        update_sql = f"""
            UPDATE features f
            SET
                rul_seconds = s.rul_seconds,
                rul_hours = s.rul_hours,
                failed = s.failed,
                censored = s.censored
            FROM {staging_table} s
            WHERE f.feature_id = s.feature_id
        """
        result = conn.execute(text(update_sql))
        logger.info(f"Updated {result.rowcount} rows in features table")

        # Drop staging table
        conn.execute(text(f"DROP TABLE {staging_table}"))

    logger.info("✓ DB updates applied successfully")


def verify_labels(engine):
    """Run verification queries and print summary"""
    logger.info("Running verification queries")

    with engine.connect() as conn:
        # Total rows
        total = conn.execute(text("SELECT COUNT(*) FROM features")).scalar()

        # Failed bearings
        failed_count = conn.execute(
            text("SELECT COUNT(*) FROM features WHERE failed = TRUE")
        ).scalar()

        # Censored bearings
        censored_count = conn.execute(
            text("SELECT COUNT(*) FROM features WHERE censored = TRUE")
        ).scalar()

        # RUL stats for failed bearings
        rul_stats = pd.read_sql(
            """
            SELECT 
                bearing_id,
                COUNT(*) as n_rows,
                ROUND(MIN(rul_hours)::numeric, 2) as min_rul_hours,
                ROUND(MAX(rul_hours)::numeric, 2) as max_rul_hours,
                ROUND(AVG(rul_hours)::numeric, 2) as avg_rul_hours
            FROM features
            WHERE failed = TRUE
            GROUP BY bearing_id
            ORDER BY bearing_id
            """,
            con=engine
        )

        logger.info(f"Total rows: {total}")
        logger.info(f"Failed: {failed_count}")
        logger.info(f"Censored: {censored_count}")

        print("\n" + "=" * 60)
        print("VERIFICATION SUMMARY")
        print("=" * 60)
        print(f"Total rows:       {total}")
        print(f"Failed bearings:  {failed_count}")
        print(f"Censored bearings: {censored_count}")
        print("\nRUL Statistics (Failed Bearings):")
        print(rul_stats.to_string(index=False))
        print("=" * 60 + "\n")

        # Sanity check
        assert failed_count + censored_count == total, "❌ Failed + Censored != Total!"
        logger.info("✓ Sanity check passed: failed + censored = total")


def main():
    engine = create_engine(DATABASE_URL, echo=False)

    # Test connection
    try:
        with engine.connect() as conn:
            db_name = conn.execute(text("SELECT current_database()")).scalar()
            logger.info(f"✓ Connected to database: {db_name}")
    except Exception as e:
        logger.exception("❌ DB connection failed: %s", e)
        raise

    # Load features
    features_df = load_features(engine)

    # Compute labels
    updates_df = compute_rul(features_df, failed_bearings={3, 4})
    logger.info(f"Prepared {len(updates_df)} update rows")

    # Choose update method based on dataset size
    if len(updates_df) < 10000:
        # Small dataset: use batch UPDATE (simpler, no staging table)
        apply_updates_batch(engine, updates_df, batch_size=1000)
    else:
        # Large dataset: use staging table (faster bulk operation)
        apply_updates_staging(engine, updates_df)

    # Verify results
    verify_labels(engine)

    logger.info("✓✓✓ Labeling complete! ✓✓✓")


if __name__ == "__main__":
    main()
