import os
import pandas as pd

DISTRESSES = ["alligator", "longitudinal", "transverse", "pothole"]

severity_levels = ["low", "medium", "high"]

severity_map = {
    "low": 0,
    "medium": 1,
    "high": 2
}

def load_astm_curve(distress, severity, base_path="../ASTM_DIGITIZED_DATA"):

    file_name = f"{distress}_{severity}.csv"
    file_path = os.path.join(base_path, file_name)

    df = pd.read_csv(file_path)

    return df

