"""
IMS Bearing Data Parser
Validates and parses raw ASCII files, handles resampling if needed.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class IMSParser:
    """Production-grade parser for IMS bearing ASCII files"""

    EXPECTED_CHANNELS = 8
    EXPECTED_FS = 20000
    TARGET_FS = 20000  # Set to 12000 if resampling needed

    @staticmethod
    def parse_timestamp(filename: str) -> datetime:
        """
        Extract timestamp from IMS filename.
        Format: YYYY.MM.DD.HH.MM.SS
        Example: 2003.10.22.12.06.24
        """
        try:
            stem = Path(filename).stem
            parts = stem.split('.')
            if len(parts) != 6:
                raise ValueError(f"Invalid filename format: {filename}")

            year, month, day, hour, minute, second = map(int, parts)
            return datetime(year, month, day, hour, minute, second)
        except Exception as e:
            logger.error(f"Failed to parse timestamp from {filename}: {e}")
            raise

    @staticmethod
    def load_file(filepath: Path, dtype=np.float32) -> np.ndarray:
        """
        Load IMS ASCII file into numpy array.
        Returns: (n_samples, 8) array
        """
        try:
            data = np.loadtxt(filepath, dtype=dtype, delimiter='\t')

            # Validate shape
            if data.ndim != 2:
                raise ValueError(f"Expected 2D array, got {data.ndim}D")
            if data.shape[1] != IMSParser.EXPECTED_CHANNELS:
                raise ValueError(
                    f"Expected {IMSParser.EXPECTED_CHANNELS} channels, "
                    f"got {data.shape[1]}"
                )

            logger.info(
                f"Loaded {filepath.name}: "
                f"{data.shape[0]} samples × {data.shape[1]} channels"
            )
            return data

        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}")
            raise

    @staticmethod
    def resample_if_needed(
            data: np.ndarray,
            source_fs: int,
            target_fs: int
    ) -> np.ndarray:
        """
        Resample data if target_fs != source_fs.
        Uses scipy.signal.resample for frequency-domain resampling.
        """
        if source_fs == target_fs:
            return data

        from scipy.signal import resample

        n_samples_original = data.shape[0]
        n_samples_target = int(n_samples_original * target_fs / source_fs)

        logger.info(
            f"Resampling from {source_fs} Hz to {target_fs} Hz "
            f"({n_samples_original} → {n_samples_target} samples)"
        )

        # Resample each channel
        resampled = np.zeros((n_samples_target, data.shape[1]), dtype=data.dtype)
        for ch in range(data.shape[1]):
            resampled[:, ch] = resample(data[:, ch], n_samples_target)

        return resampled

    @staticmethod
    def validate_file(filepath: Path) -> Tuple[bool, str]:
        """
        Pre-flight validation without loading full file.
        Returns: (is_valid, error_message)
        """
        if not filepath.exists():
            return False, f"File not found: {filepath}"

        if filepath.stat().st_size == 0:
            return False, f"Empty file: {filepath}"

        # Quick check: read first line
        try:
            with open(filepath, 'r') as f:
                first_line = f.readline().strip()
                cols = first_line.split('\t')
                if len(cols) != IMSParser.EXPECTED_CHANNELS:
                    return False, f"Expected 8 columns, got {len(cols)}"
        except Exception as e:
            return False, f"Read error: {e}"

        return True, "OK"

    @classmethod
    def parse_file(
            cls,
            filepath: Path,
            resample_to: Optional[int] = None
    ) -> Tuple[np.ndarray, datetime, dict]:
        """
        Full parse pipeline: validate → load → resample → return.

        Returns:
            data: (n_samples, 8) array
            timestamp: datetime of recording
            metadata: dict with fs, n_samples, etc.
        """
        # Validate
        is_valid, msg = cls.validate_file(filepath)
        if not is_valid:
            raise ValueError(msg)

        # Parse timestamp
        timestamp = cls.parse_timestamp(filepath.name)

        # Load
        data = cls.load_file(filepath)

        # Resample if requested
        source_fs = cls.EXPECTED_FS
        if resample_to is not None and resample_to != source_fs:
            data = cls.resample_if_needed(data, source_fs, resample_to)
            final_fs = resample_to
        else:
            final_fs = source_fs

        metadata = {
            'filename': filepath.name,
            'timestamp': timestamp,
            'fs': final_fs,
            'n_samples': data.shape[0],
            'n_channels': data.shape[1],
            'duration_sec': data.shape[0] / final_fs
        }

        return data, timestamp, metadata


# Quick CLI test
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python parse.py <path_to_ims_file>")
        sys.exit(1)

    logging.basicConfig(level=logging.INFO)
    filepath = Path(sys.argv[1])

    data, ts, meta = IMSParser.parse_file(filepath)
    print(f"\nParsed: {meta['filename']}")
    print(f"Timestamp: {ts}")
    print(f"Shape: {data.shape}")
    print(f"Duration: {meta['duration_sec']:.3f} sec")
    print(f"First 5 samples, Channel 0:\n{data[:5, 0]}")
