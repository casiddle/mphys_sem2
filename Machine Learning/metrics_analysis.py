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

data_set=r"training_metrics_3_targets.csv"
df = pd.read_csv(data_set)
print(df.describe(include='all'))
df = df.drop(columns=['loss_function','activation_function','predicted_feature','overfitting','optimiser','test_size'])

correlation_kendall = df.corr(method='kendall')

# Plot Kendall correlation matrix with bwr colormap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_kendall, annot=True, cmap="bwr", fmt=".2f", linewidths=0.5)
plt.title("Kendall Correlation Matrix")
plt.show()

#Rank correlations
# Convert the correlation matrix into a long format
correlation_long = correlation_kendall.unstack().reset_index()
# Rename columns for clarity
correlation_long.columns = ['Feature 1', 'Feature 2', 'Correlation']
# Remove self-correlations (where Feature 1 == Feature 2)
correlation_long = correlation_long[correlation_long['Feature 1'] != correlation_long['Feature 2']]
# Drop duplicate pairs (each correlation appears twice in a symmetric matrix)
correlation_long['Sorted Pair'] = correlation_long.apply(lambda x: tuple(sorted([x['Feature 1'], x['Feature 2']])), axis=1)
correlation_long = correlation_long.drop_duplicates(subset=['Sorted Pair']).drop(columns=['Sorted Pair'])
# Sort by absolute correlation value (strongest relationships first)
correlation_long = correlation_long.reindex(correlation_long['Correlation'].abs().sort_values(ascending=False).index)
# Display the ranked correlation pairs
print(correlation_long)
# Save the ranked correlation pairs to a CSV file
correlation_long.to_csv('metrics_correlation_ranked.csv', index=False)

# Filter rows where 'Feature 1' or 'Feature 2' contains 'Emittance'
test_loss_filtered_df = correlation_long[(correlation_long["Feature 1"]=="test_loss") | (correlation_long["Feature 2"]=="test_loss")]
print(test_loss_filtered_df.head(15))



