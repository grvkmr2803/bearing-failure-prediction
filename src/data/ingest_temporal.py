# src/data/ingest_temporal.py
"""
Ingest temporal features parquet into PostgreSQL
Handles 220+ columns dynamically
"""
import os
import sys
import logging
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text, inspect

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('IngestTemporal')

DATABASE_URL = os.getenv(
    'DATABASE_URL',
    'postgresql://postgres:postgres@localhost:5432/anudeep'
)


def get_existing_columns(engine) -> set:
    """Get existing column names from features table"""
    inspector = inspect(engine)
    columns = [col['name'] for col in inspector.get_columns('features')]
    return set(columns)


def add_missing_columns(engine, df: pd.DataFrame):
    """Dynamically add missing columns to features table"""
    existing_cols = get_existing_columns(engine)
    df_cols = set(df.columns)

    # Find columns in parquet but not in DB
    missing_cols = df_cols - existing_cols

    if not missing_cols:
        logger.info("✓ All columns already exist in database")
        return

    logger.info(f"Adding {len(missing_cols)} missing columns to database...")

    with engine.begin() as conn:
        for col in missing_cols:
            # Skip metadata columns
            if col in ['file', 'timestamp', 'channel', 'bearing', 'axis']:
                continue

            # Determine SQL type based on pandas dtype
            dtype = df[col].dtype
            if dtype in ['int64', 'int32']:
                sql_type = 'INTEGER'
            elif dtype in ['float64', 'float32']:
                sql_type = 'REAL'
            elif dtype == 'bool':
                sql_type = 'BOOLEAN'
            else:
                sql_type = 'TEXT'

            try:
                conn.execute(text(
                    f'ALTER TABLE features ADD COLUMN IF NOT EXISTS "{col}" {sql_type}'
                ))
                logger.info(f"  ✓ Added column: {col} ({sql_type})")
            except Exception as e:
                logger.warning(f"  ⚠ Failed to add {col}: {e}")


def ingest_temporal_features(parquet_path: Path, table_name: str = 'features'):
    """
    Load temporal features parquet and merge with existing features table
    """
    logger.info("=" * 60)
    logger.info("INGESTING TEMPORAL FEATURES")
    logger.info("=" * 60)

    # Load parquet
    logger.info(f"Loading: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    logger.info(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")

    # Rename columns to match DB schema
    column_mapping = {
        'file': 'file_name',
        'bearing': 'bearing_id'
    }
    df = df.rename(columns=column_mapping)

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Connect to DB
    engine = create_engine(DATABASE_URL, echo=False)
    logger.info("✓ Connected to database")

    # Add missing columns dynamically
    add_missing_columns(engine, df)

    # Strategy: Instead of INSERT, we'll UPDATE existing rows
    # Match on (file_name, bearing_id, axis)

    logger.info("Updating features table with temporal features...")

    # Get feature columns (exclude keys)
    key_cols = {'file_name', 'timestamp', 'bearing_id', 'axis', 'channel'}
    feature_cols = [c for c in df.columns if c not in key_cols]

    logger.info(f"Key columns: {key_cols}")
    logger.info(f"Feature columns to update: {len(feature_cols)}")

    # Build UPDATE statement
    # We'll do this in chunks for safety
    chunk_size = 1000
    n_chunks = (len(df) - 1) // chunk_size + 1

    updated_count = 0

    with engine.begin() as conn:
        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, len(df))
            chunk_df = df.iloc[start_idx:end_idx]

            # Create temp table for this chunk
            temp_table = f'temp_features_chunk_{chunk_idx}'
            chunk_df.to_sql(
                temp_table,
                con=engine,
                if_exists='replace',
                index=False,
                method='multi'
            )

            # Build SET clause dynamically
            set_clause = ', '.join([
                f'"{col}" = t."{col}"' for col in feature_cols if col in chunk_df.columns
            ])

            # Update main table from temp
            update_sql = f"""
                UPDATE features f
                SET {set_clause}
                FROM {temp_table} t
                WHERE f.file_name = t.file_name
                  AND f.bearing_id = t.bearing_id
                  AND f.axis = t.axis
            """

            result = conn.execute(text(update_sql))
            updated_count += result.rowcount

            # Drop temp table
            conn.execute(text(f'DROP TABLE {temp_table}'))

            logger.info(f"  Chunk {chunk_idx + 1}/{n_chunks}: Updated {result.rowcount} rows")

    logger.info("=" * 60)
    logger.info(f"✓✓✓ INGEST COMPLETE ✓✓✓")
    logger.info(f"✓ Updated {updated_count} rows with temporal features")
    logger.info("=" * 60)

    # Verify
    with engine.connect() as conn:
        sample = pd.read_sql(
            """
            SELECT file_name, bearing_id, rms_mean, rms_mean_roll_5_mean, rms_mean_slope_5
            FROM features
            WHERE rms_mean_roll_5_mean IS NOT NULL
            LIMIT 5
            """,
            con=conn
        )
        logger.info("\nSample rows with temporal features:")
        print(sample)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python src/data/ingest_temporal.py <path_to_temporal_parquet>")
        sys.exit(1)

    parquet_path = Path(sys.argv[1])

    if not parquet_path.exists():
        print(f"Error: File not found: {parquet_path}")
        sys.exit(1)

    ingest_temporal_features(parquet_path)
