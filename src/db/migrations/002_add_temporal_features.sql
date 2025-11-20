-- Materialized view for failed bearings
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_failed_bearings_features AS
SELECT 
    feature_id,
    file_name,
    timestamp,
    bearing_id,
    axis,
    rul_hours,
    rms_mean,
    kurtosis_mean,
    crest_factor_mean
FROM features
WHERE failed = TRUE
ORDER BY bearing_id, axis, timestamp;

CREATE INDEX IF NOT EXISTS idx_mv_failed_bearing_ts 
ON mv_failed_bearings_features(bearing_id, timestamp);

CREATE INDEX IF NOT EXISTS idx_mv_failed_rul 
ON mv_failed_bearings_features(rul_hours);

-- View for latest bearing status
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_latest_bearing_status AS
WITH latest_data AS (
    SELECT 
        bearing_id,
        axis,
        MAX(timestamp) as last_timestamp
    FROM features
    GROUP BY bearing_id, axis
)
SELECT 
    f.bearing_id,
    f.axis,
    f.timestamp,
    f.rms_mean,
    f.kurtosis_mean,
    f.rul_hours,
    f.failed,
    f.censored
FROM features f
INNER JOIN latest_data ld 
    ON f.bearing_id = ld.bearing_id 
    AND f.axis = ld.axis 
    AND f.timestamp = ld.last_timestamp;

CREATE INDEX IF NOT EXISTS idx_mv_latest_bearing 
ON mv_latest_bearing_status(bearing_id);
