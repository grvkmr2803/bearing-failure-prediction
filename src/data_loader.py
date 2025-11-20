import glob
import os
import numpy as np

def list_files(set_path: str):
    """List and chronologically sort all IMS files in a given set folder"""
    pattern = os.path.join(set_path, "*")
    files = glob.glob(pattern)
    files.sort()
    return files

def load_file(file_path: str, dtype=np.float32):
    """
    Load one IMS ASCII file into a 2D numpy array.
    Set 1 has shape (20480, 8) -> 20480 rows x 8 channels.
    """
    return np.loadtxt(file_path, dtype=dtype)

def load_channel(file_path: str, channel_index: int, dtype=np.float32):
    """
    Load one channel from a file. Each bearing has 2 channels.
    Bearing 1 -> Channel 0,1 and so on up to Bearing 4.
    """
    data = np.loadtxt(file_path, dtype=dtype)
    return data[:, channel_index]