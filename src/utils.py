import os
from datetime import datetime as dt

def parse_ims_timestamp(file_path: str):
    """
    Extract pandas/py datetime from IMS filename 'YYYY.MM.DD.HH.MM.SS'.
    """
    name = os.path.basename(file_path)
    ts = dt.strptime(name, "%Y.%m.%d.%H.%M.%S")
    return ts