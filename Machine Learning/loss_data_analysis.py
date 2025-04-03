import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



data = pd.read_csv(r'Machine Learning\loss_data.csv')  

point_array=np.array([100,200,300,400,500,600,1000,2000,2400])
# Group by 'No. Data Points' and calculate the mean of the losses
average_losses = data.groupby('No. Data Points').mean()
average_losses.reset_index(inplace=True)
#average_losses = average_losses[(average_losses['No. Data Points'] != 575) & (average_losses['No. Data Points']>=100)]
average_losses=average_losses[average_losses['No. Data Points'].isin(point_array)]

# Display the result
print(average_losses)

# Create a figure with 3 subplots (1 row, 3 columns)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # 1 row, 3 columns
# Plotting each loss array in its respective subplot
axes[0].plot(average_losses['No. Data Points'], average_losses['Emittance Loss'], label='Emittance Loss', color='b', linestyle='-', marker='o')
axes[0].set_title('Emittance Loss', fontsize=14)
axes[0].set_xlabel('No. Data points', fontsize=12)
axes[0].set_ylabel('Loss Value', fontsize=12)
axes[0].grid(True)
axes[0].legend()

axes[1].plot(average_losses['No. Data Points'], average_losses['Energy Loss'], label='Energy Loss', color='g', linestyle='--', marker='x')
axes[1].set_title('Energy Loss', fontsize=14)
axes[1].set_xlabel('No. Data points', fontsize=12)
axes[1].set_ylabel('Loss Value', fontsize=12)
axes[1].grid(True)
axes[1].legend()

axes[2].plot(average_losses['No. Data Points'], average_losses['Spread Loss'], label='Spread Loss', color='r', linestyle='-.', marker='s')
axes[2].set_title('Spread Loss', fontsize=14)
axes[2].set_xlabel('No. Data points', fontsize=12)
axes[2].set_ylabel('Loss Value', fontsize=12)
axes[2].grid(True)
axes[2].legend()

# Adjust layout to prevent overlapping of titles and labels
plt.tight_layout()
plt.savefig(r'Machine Learning\loss_data_analysis_plot.png')

# Showing the plot
plt.show()
