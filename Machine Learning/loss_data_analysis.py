import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



data = pd.read_csv(r'Machine Learning\loss_data.csv')  

point_array=np.array([100,200,300,400,500,600,1000,2000,2400])
# Group by 'No. Data Points' and calculate the mean of the losses
average_losses = data.groupby('No. Data Points').agg(['mean', 'var'])
average_losses.reset_index(inplace=True)
average_losses.reset_index(inplace=True)
#average_losses = average_losses[(average_losses['No. Data Points'] != 575) & (average_losses['No. Data Points']>=100)]
average_losses=average_losses[average_losses['No. Data Points'].isin(point_array)]

# Display the result
print(average_losses)

# Create a figure with 3 subplots (1 row, 3 columns)
fig, axes = plt.subplots(1, 3, figsize=(16, 6))  # 1 row, 3 columns
# Plotting each loss array in its respective subplot
#axes[2].plot(average_losses['No. Data Points'], average_losses[("Emittance Loss", "mean")], label='Emittance Loss', color='tab:purple', linestyle='-', marker='o')
axes[2].errorbar(
    average_losses['No. Data Points'],  # X values
    average_losses[("Emittance Loss", "mean")],  # Y values (mean)
    yerr=np.sqrt(average_losses[("Emittance Loss", "var")]),  # Error bars (std dev)
    label='Emittance Loss',
    color='tab:purple',
    linestyle='-.',
    marker='s',
    capsize=5  # Adds caps to the error bars for better visibility
)
axes[2].set_title(r'Emittance Loss ($\mu m$)', fontsize=14)
axes[2].set_xlabel('Number of Data points', fontsize=12)
axes[2].set_ylabel(r'Loss ($\mu m$)', fontsize=12)
axes[2].grid(True)
axes[2].legend()

#axes[1].plot(average_losses['No. Data Points'], average_losses[("Energy Loss", "mean")], label='Energy Loss', color='tab:green', linestyle='--', marker='x')
axes[1].errorbar(
    average_losses['No. Data Points'],  # X values
    average_losses[("Energy Loss", "mean")],  # Y values (mean)
    yerr=np.sqrt(average_losses[("Energy Loss", "var")]),  # Error bars (std dev)
    label='Energy Loss',
    color='tab:green',
    linestyle='-.',
    marker='s',
    capsize=5  # Adds caps to the error bars for better visibility
)
axes[1].set_title('Energy Loss (GeV)', fontsize=14)
axes[1].set_xlabel('Number of Data points', fontsize=12)
axes[1].set_ylabel('Loss (GeV)', fontsize=12)
axes[1].grid(True)
axes[1].legend()

axes[0].errorbar(
    average_losses['No. Data Points'],  # X values
    average_losses[("Spread Loss", "mean")],  # Y values (mean)
    yerr=np.sqrt(average_losses[("Spread Loss", "var")]),  # Error bars (std dev)
    label='Spread Loss',
    color='tab:blue',
    linestyle='-.',
    marker='s',
    capsize=5  # Adds caps to the error bars for better visibility
)
axes[0].set_title('Spread Loss (%)', fontsize=14)
axes[0].set_xlabel('Number of Data points', fontsize=12)
axes[0].set_ylabel('Loss (%)', fontsize=12)
axes[0].grid(True)
axes[0].legend()

# Adjust layout to prevent overlapping of titles and labels
plt.tight_layout()
plt.savefig(r'Machine Learning\loss_data_analysis_plot.png')

# Showing the plot
plt.show()
