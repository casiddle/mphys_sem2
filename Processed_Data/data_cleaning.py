import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize
import os

# Get the directory where the current script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Change the current working directory to the script directory
os.chdir(script_dir)

def data_frame_cleaning(data_frame):
  data_frame.dropna(inplace=True)
  num_duplicates = data_frame.duplicated().sum()
  print(f"Number of duplicate rows: {num_duplicates}")
  data_frame.drop_duplicates(inplace=True)
  print("Cleaning Complete")
  return data_frame

data_set=r"data_sets\output_test_96.csv"
df = pd.read_csv(data_set)
df2=pd.read_csv(data_set)
#Check for missing values-----------------------------------------------------------------------------
print(df.isna())    # Detect missing values
print(df.isna().sum())  # Count missing values in each column

df.dropna(inplace=True)  # Drops rows with any NaN

#Check for duplicate lines-------------------------------------------------------------------------------
# Count total duplicates
num_duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {num_duplicates}")

# Drop duplicates
df.drop_duplicates(inplace=True)

data_frame_cleaning(df2)