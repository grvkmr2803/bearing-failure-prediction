# Quick sanity check on your DB
import pandas as pd
from sqlalchemy import create_engine, text

engine = create_engine("postgresql://postgres:postgres@localhost:5432/anudeep")

# Check feature completeness
with engine.connect() as conn:
    # Null counts per column
    null_check = pd.read_sql("""
        SELECT 
            COUNT(*) FILTER (WHERE rms_mean IS NULL) as null_rms,
            COUNT(*) FILTER (WHERE kurtosis_mean IS NULL) as null_kurtosis,
            COUNT(*) FILTER (WHERE rul_hours IS NULL) as null_rul,
            COUNT(*) as total_rows
        FROM features
    """, con=conn)
    print("Null Check:")
    print(null_check)

    # Feature ranges (detect outliers)
    ranges = pd.read_sql("""
        SELECT 
            ROUND(MIN(rms_mean)::numeric, 4) as min_rms,
            ROUND(MAX(rms_mean)::numeric, 4) as max_rms,
            ROUND(AVG(rms_mean)::numeric, 4) as avg_rms,
            ROUND(MIN(kurtosis_mean)::numeric, 2) as min_kurt,
            ROUND(MAX(kurtosis_mean)::numeric, 2) as max_kurt
        FROM features
        WHERE failed = TRUE
    """, con=conn)
    print("\nFailed Bearings - Feature Ranges:")
    print(ranges)
