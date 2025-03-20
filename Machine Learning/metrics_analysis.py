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

data_set=r"training_metrics_3_targets.csv"
df = pd.read_csv(data_set)
df=data_frame_cleaning(df)
print(df.describe(include='all'))
df = df.drop(columns=['loss_function','activation_function','predicted_feature','predictor_features','overfitting','optimiser','test_size'])

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



data_set_2=r"training_metrics_3_targets_no_nodes.csv"
df2 = pd.read_csv(data_set_2)
df2=data_frame_cleaning(df2)

df2 = df2.sort_values('no_nodes')


# Create a figure and axis
fig, ax1 = plt.subplots()

# Plot Loss on the primary y-axis
ax1.plot(df2['no_nodes'], df2['test_loss'], color='r', label='Test Loss')
ax1.plot(df2['no_nodes'], df2['Emittance_test_loss'], color='k', label='Emittance Test Loss')
ax1.plot(df2['no_nodes'], df2['Beam Energy_test_loss'], color='blue', label='Beam Energy Test Loss')
ax1.plot(df2['no_nodes'], df2['Beam Spread_test_loss'], color='green', label='Beam Spread Test Loss')

# Axis labels for Loss
ax1.set_xlabel('Number of Nodes')
ax1.set_ylabel('Loss', color='r')
ax1.tick_params(axis='y', labelcolor='r')
ax1.set_yscale('log')  # Optional log scale for better visibility

# Create the secondary y-axis for Accuracy
ax2 = ax1.twinx()

# Plot Accuracy on the secondary y-axis
ax2.plot(df2['no_nodes'], df2['Emittance_test_accuracy'], color='k', linestyle='--', label='Emittance Test Accuracy')
ax2.plot(df2['no_nodes'], df2['Beam Energy_test_accuracy'], color='blue', linestyle='--', label='Beam Energy Test Accuracy')
ax2.plot(df2['no_nodes'], df2['Beam Spread_test_accuracy'], color='green', linestyle='--', label='Beam Spread Test Accuracy')

# Axis labels for Accuracy
ax2.set_ylabel('Accuracy', color='b')
ax2.tick_params(axis='y', labelcolor='b')

# Combining legends from both axes
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper right')

plt.title('Test Loss and Accuracy vs Number of Nodes')
#plt.show()




data_set_3=r"training_metrics_3_targets_no_layers.csv"
df3 = pd.read_csv(data_set_3)
df3=data_frame_cleaning(df3)

df3 = df3.sort_values('no_hidden_layers')


# Create a figure and axis
fig, ax1 = plt.subplots()

# Plot Loss on the primary y-axis
ax1.plot(df3['no_hidden_layers'], df3['test_loss'], color='r', label='Test Loss')
ax1.plot(df3['no_hidden_layers'], df3['Emittance_test_loss'], color='k', label='Emittance Test Loss')
ax1.plot(df3['no_hidden_layers'], df3['Beam Energy_test_loss'], color='blue', label='Beam Energy Test Loss')
ax1.plot(df3['no_hidden_layers'], df3['Beam Spread_test_loss'], color='green', label='Beam Spread Test Loss')

# Axis labels for Loss
ax1.set_xlabel('Number of Hidden Layers')
ax1.set_ylabel('Loss', color='r')
ax1.tick_params(axis='y', labelcolor='r')
ax1.set_yscale('log')  # Optional log scale for better visibility

# Create the secondary y-axis for Accuracy
ax2 = ax1.twinx()

# Plot Accuracy on the secondary y-axis
ax2.plot(df3['no_hidden_layers'], df3['Emittance_test_accuracy'], color='k', linestyle='--', label='Emittance Test Accuracy')
ax2.plot(df3['no_hidden_layers'], df3['Beam Energy_test_accuracy'], color='blue', linestyle='--', label='Beam Energy Test Accuracy')
ax2.plot(df3['no_hidden_layers'], df3['Beam Spread_test_accuracy'], color='green', linestyle='--', label='Beam Spread Test Accuracy')

# Axis labels for Accuracy
ax2.set_ylabel('Accuracy', color='b')
ax2.tick_params(axis='y', labelcolor='b')

# Combining legends from both axes
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper right')

plt.title('Test Loss and Accuracy vs Number of Hidden Layers')
#plt.show()


data_set_4=r"training_metrics_3_targets_learning_rate.csv"
df4 = pd.read_csv(data_set_4)
df4=data_frame_cleaning(df4)

df4 = df4.sort_values('learning_rate')


# Create a figure and axis
fig, ax1 = plt.subplots()

# Plot Loss on the primary y-axis
ax1.plot(df4['learning_rate'], df4['test_loss'], color='r', label='Test Loss')
ax1.plot(df4['learning_rate'], df4['Emittance_test_loss'], color='k', label='Emittance Test Loss')
ax1.plot(df4['learning_rate'], df4['Beam Energy_test_loss'], color='blue', label='Beam Energy Test Loss')
ax1.plot(df4['learning_rate'], df4['Beam Spread_test_loss'], color='green', label='Beam Spread Test Loss')

# Axis labels for Loss
ax1.set_xlabel('Number of learning rate')
ax1.set_ylabel('Loss', color='r')
ax1.tick_params(axis='y', labelcolor='r')
ax1.set_yscale('log')  # Optional log scale for better visibility
ax1.set_xscale('log')

# Create the secondary y-axis for Accuracy
ax2 = ax1.twinx()

# Plot Accuracy on the secondary y-axis
ax2.plot(df4['learning_rate'], df4['Emittance_test_accuracy'], color='k', linestyle='--', label='Emittance Test Accuracy')
ax2.plot(df4['learning_rate'], df4['Beam Energy_test_accuracy'], color='blue', linestyle='--', label='Beam Energy Test Accuracy')
ax2.plot(df4['learning_rate'], df4['Beam Spread_test_accuracy'], color='green', linestyle='--', label='Beam Spread Test Accuracy')

# Axis labels for Accuracy
ax2.set_ylabel('Accuracy', color='b')
ax2.tick_params(axis='y', labelcolor='b')

# Combining legends from both axes
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper right')

plt.title('Test Loss and Accuracy vs Learning Rate')
plt.show()