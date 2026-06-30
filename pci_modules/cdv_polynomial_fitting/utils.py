import os
import pandas as pd

base_path = os.path.join(os.getcwd(), "pci_modules", "ASTM_DIGITIZED_DATA", "cdv")
# curve_files = os.listdir(base_path)
# print(curve_files)
file_name = f"7.csv"

# print(file_path)
file_path = os.path.join(base_path, file_name)

df = pd.read_csv(file_path, header=None)

df[0] = df[0].round(6)
tdv = df[0].tolist()
print("tdv \n")
print(tdv)
print("\n \n")

df[1] = df[1].round(6)
cdv = df[1].tolist()
print("cdv \n")
print(cdv)