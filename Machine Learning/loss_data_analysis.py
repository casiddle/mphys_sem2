import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



data = pd.read_csv(r'Machine Learning\loss_data.csv')

# Apply square root transformation to the first three columns
data.iloc[:, :3] = np.sqrt(data.iloc[:, :3])

point_array=np.array([500,600,700,800,1000,1200,1400,1600,1800,2000,2400])
# Group by 'No. Data Points' and calculate the mean of the losses
average_losses = data.groupby('No. Data Points').agg(['mean', 'var'])
average_losses.reset_index(inplace=True)
average_losses.reset_index(inplace=True)
#average_losses = average_losses[(average_losses['No. Data Points'] != 575) & (average_losses['No. Data Points']>=100)]
average_losses=average_losses[average_losses['No. Data Points'].isin(point_array)]

# Display the result
print(average_losses)

# # Create a figure with 3 subplots (1 row, 3 columns)
# fig, axes = plt.subplots(1, 3, figsize=(16, 6))  # 1 row, 3 columns
# # Plotting each loss array in its respective subplot
# #axes[2].plot(average_losses['No. Data Points'], average_losses[("Emittance Loss", "mean")], label='Emittance Loss', color='tab:purple', linestyle='-', marker='o')
# axes[2].errorbar(
#     average_losses['No. Data Points'],  # X values
#     average_losses[("Emittance Loss", "mean")],  # Y values (mean)
#     yerr=np.sqrt(average_losses[("Emittance Loss", "var")]),  # Error bars (std dev)
#     label='Emittance Loss',
#     color='tab:purple',
#     linestyle='-.',
#     marker='s',
#     capsize=5  # Adds caps to the error bars for better visibility
# )
# axes[2].set_title(r'Emittance Loss ($\mu m$)', fontsize=14)
# axes[2].set_xlabel('Number of Data points', fontsize=12)
# axes[2].set_ylabel(r'Loss ($\mu m$)', fontsize=12)
# axes[2].grid(True)
# axes[2].legend()

# #axes[1].plot(average_losses['No. Data Points'], average_losses[("Energy Loss", "mean")], label='Energy Loss', color='tab:green', linestyle='--', marker='x')
# axes[1].errorbar(
#     average_losses['No. Data Points'],  # X values
#     average_losses[("Energy Loss", "mean")],  # Y values (mean)
#     yerr=np.sqrt(average_losses[("Energy Loss", "var")]),  # Error bars (std dev)
#     label='Energy Loss',
#     color='tab:green',
#     linestyle='-.',
#     marker='s',
#     capsize=5  # Adds caps to the error bars for better visibility
# )
# axes[1].set_title('Energy Loss (GeV)', fontsize=14)
# axes[1].set_xlabel('Number of Data points', fontsize=12)
# axes[1].set_ylabel('Loss (GeV)', fontsize=12)
# axes[1].grid(True)
# axes[1].legend()

# axes[0].errorbar(
#     average_losses['No. Data Points'],  # X values
#     average_losses[("Spread Loss", "mean")],  # Y values (mean)
#     yerr=np.sqrt(average_losses[("Spread Loss", "var")]),  # Error bars (std dev)
#     label='Spread Loss',
#     color='tab:blue',
#     linestyle='-.',
#     marker='s',
#     capsize=5  # Adds caps to the error bars for better visibility
# )
# axes[0].set_title('Spread Loss (%)', fontsize=14)
# axes[0].set_xlabel('Number of Data points', fontsize=12)
# axes[0].set_ylabel('Loss (%)', fontsize=12)
# axes[0].grid(True)
# axes[0].legend()

# # Adjust layout to prevent overlapping of titles and labels
# plt.tight_layout()
# plt.savefig(r'Machine Learning\loss_data_analysis_plot.png')

# Showing the plot
plt.show()



# Create a figure with 3 subplots (1 row, 3 columns)
fig, axes = plt.subplots(1, 3, figsize=(16, 6))  # 1 row, 3 columns

# Plot variance of Spread Loss
axes[0].plot(
    average_losses['No. Data Points'],  # X values
    average_losses[("Spread Loss", "var")],  # Y values (variance)
    label='Spread Loss Variance',
    color='tab:blue',
    linestyle='-',
    marker='o'
)
axes[0].set_title('Spread Loss Variance', fontsize=14)
axes[0].set_xlabel('Number of Data Points', fontsize=12)
axes[0].set_ylabel('Variance', fontsize=12)
axes[0].grid(True)
axes[0].legend()

# Plot variance of Energy Loss
axes[1].plot(
    average_losses['No. Data Points'],  
    average_losses[("Energy Loss", "var")],  
    label='Energy Loss Variance',
    color='tab:green',
    linestyle='-',
    marker='o'
)
axes[1].set_title('Energy Loss Variance', fontsize=14)
axes[1].set_xlabel('Number of Data Points', fontsize=12)
axes[1].set_ylabel('Variance', fontsize=12)
axes[1].grid(True)
axes[1].legend()

# Plot variance of Emittance Loss
axes[2].plot(
    average_losses['No. Data Points'],  
    average_losses[("Emittance Loss", "var")],  
    label='Emittance Loss Variance',
    color='tab:purple',
    linestyle='-',
    marker='o'
)
axes[2].set_title(r'Emittance Loss Variance', fontsize=14)
axes[2].set_xlabel('Number of Data Points', fontsize=12)
axes[2].set_ylabel('Variance', fontsize=12)
axes[2].grid(True)
axes[2].legend()

# Adjust layout to prevent overlapping
plt.tight_layout()

# Save the figure
plt.savefig(r'Machine Learning\variance_plot.png', dpi=300)

# Show the plot
plt.show()



# Increase font sizes globally
plt.rcParams.update({
    "font.size": 14,           # Default font size
    "axes.titlesize": 22,      # Title font size
    "axes.labelsize": 20,      # X and Y label font size
    "xtick.labelsize": 18,     # X-axis tick font size
    "ytick.labelsize": 18,     # Y-axis tick font size
    "legend.fontsize": 20      # Legend font size
})

# Function to create error bar plots in separate windows
def create_errorbar_plot(x, y, yerr, x_label, y_label, legend, title, color='tab:blue'):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(x, y, yerr=yerr, label=legend, color=color, linestyle='-.', marker='s', capsize=5)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True)
    ax.legend()
    plt.show()

# Individual plots
create_errorbar_plot(
    average_losses['No. Data Points'], 
    average_losses[("Spread Loss", "mean")], 
    np.sqrt(average_losses[("Spread Loss", "var")]), 
    'Number of Data Points', 
    'RMSE (%)', 
    'Beam Spread RMSE',
    'Beam Spread error vs no. points in training data', 
    'tab:blue'
)

create_errorbar_plot(
    average_losses['No. Data Points'], 
    average_losses[("Energy Loss", "mean")], 
    np.sqrt(average_losses[("Energy Loss", "var")]), 
    'Number of Data Points', 
    'RMSE (GeV)', 
    'Beam Energy RMSE', 
    'Beam Energy error vs no. points in training data', 
    'tab:green'
)

create_errorbar_plot(
    average_losses['No. Data Points'], 
    average_losses[("Emittance Loss", "mean")], 
    np.sqrt(average_losses[("Emittance Loss", "var")]), 
    'Number of Data Points', 
    r'RMSE ($\mu m$)', 
    'Emittance RMSE', 
    'Emittance error vs no. points in training data', 
    'tab:purple'
)