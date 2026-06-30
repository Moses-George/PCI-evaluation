import os
import pandas as pd

base_path = os.path.join(os.getcwd(), "pci_modules", "ASTM_DIGITIZED_DATA")
# curve_files = os.listdir(base_path)
# print(curve_files)
file_name = f"pothole_high.csv"

# print(file_path)
file_path = os.path.join(base_path, file_name)

df = pd.read_csv(file_path, header=None)

df[0] = df[0].round(6)
density = df[0].tolist()
print(density)

df[1] = df[1].round(6)
dv = df[1].tolist()
print(dv)
