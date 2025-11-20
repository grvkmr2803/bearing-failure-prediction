from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR/"data"
PROCESSED_DIR = DATA_DIR/"processed"
SET1_FEATURES = PROCESSED_DIR/"set1_features.parquet"
SET1_LABELED = PROCESSED_DIR/"set1_labeled.parquet"


SET1_CHANNEL_MAP = {
    0: {"bearing": 1, "axis": "x"},
    1: {"bearing": 1, "axis": "y"},
    2: {"bearing": 2, "axis": "x"},
    3: {"bearing": 2, "axis": "y"},
    4: {"bearing": 3, "axis": "x"},
    5: {"bearing": 3, "axis": "y"},
    6: {"bearing": 4, "axis": "x"},
    7: {"bearing": 4, "axis": "y"},
}

FS = 20000      # sampling rate (Hz)
WIN = 2048      # samples/window (~0.1024 s)
OVERLAP = 0.5   # 50%