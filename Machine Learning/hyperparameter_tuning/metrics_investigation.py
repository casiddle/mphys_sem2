import subprocess
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import re

df=pd.read_csv(r'Machine Learning\hyperparameter_tuning\training_metrics_3_targets_optimisation.csv')



# Find the optimal combination that minimizes each loss
optimal_combination = {
    'test_loss': df.loc[df['test_loss'].idxmin()],
    'Emittance_test_loss': df.loc[df['Emittance_test_loss'].idxmin()],
    'Beam Energy_test_loss': df.loc[df['Beam Energy_test_loss'].idxmin()],
    'Beam Spread_test_loss': df.loc[df['Beam Spread_test_loss'].idxmin()]
}

# Display optimal combinations
print("\nOptimal Combinations:")
for loss, row in optimal_combination.items():
    print(f"{loss}: {row[['no_hidden_layers', 'no_nodes', 'learning_rate', loss]]}")

filtered_df=df[df['test_loss']<300]
print(filtered_df[['test_loss', 'no_hidden_layers','no_nodes','learning_rate','Emittance_test_loss','Beam Spread_test_loss','Beam Energy_test_loss']])

df_data=pd.read_csv(r'Processed_Data\data_sets\output_test_600.csv')
print(df_data[['Beam Spread','Beam Energy']])