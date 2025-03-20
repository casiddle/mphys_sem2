import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os


# Get the directory where the current script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Change the current working directory to the script directory
os.chdir(script_dir)

data_set=r"data_sets\output_test_600.csv"
df = pd.read_csv(data_set)
df=df[df['Beam Spread']>0.01]

filtered_df1=df[df['Set Radius']==1]#& df['Beam Spread']>0.01]
filtered_df2=df[df['Set Radius']==2]
filtered_df3=df[df['Set Radius']==3]
filtered_df4=df[df['Set Radius']==4]
filtered_df5=df[df['Set Radius']==5]
filtered_df05=df[df['Set Radius']==0.5]
print(filtered_df1[['Set Radius','Set Emittance','Beam Spread', 'Beam Energy','Emittance']])

# Create 3 subplots stacked vertically
fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

# Plot each variable on its own subplot
axes[0].plot(filtered_df1['Set Emittance'], filtered_df1['Beam Energy'], label='Matched Radius')
axes[0].plot(filtered_df2['Set Emittance'], filtered_df2['Beam Energy'], label='2*Matched Radius')
axes[0].plot(filtered_df3['Set Emittance'], filtered_df3['Beam Energy'], label='3*Matched Radius')
axes[0].plot(filtered_df4['Set Emittance'], filtered_df4['Beam Energy'], label='4*Matched Radius')
axes[0].plot(filtered_df5['Set Emittance'], filtered_df5['Beam Energy'], label='5*Matched Radius')
axes[0].plot(filtered_df05['Set Emittance'], filtered_df05['Beam Energy'], label='0.5*Matched Radius')

axes[1].plot(filtered_df1['Set Emittance'], filtered_df1['Beam Spread'], label='Matched Radius')
axes[1].plot(filtered_df2['Set Emittance'], filtered_df2['Beam Spread'], label='2*Matched Radius')
axes[1].plot(filtered_df3['Set Emittance'], filtered_df3['Beam Spread'], label='3*Matched Radius')
axes[1].plot(filtered_df4['Set Emittance'], filtered_df4['Beam Spread'], label='4*Matched Radius')
axes[1].plot(filtered_df5['Set Emittance'], filtered_df5['Beam Spread'], label='5*Matched Radius')
axes[1].plot(filtered_df05['Set Emittance'], filtered_df05['Beam Spread'], label='0.5*Matched Radius')

axes[2].plot(filtered_df1['Set Emittance'], filtered_df1['Emittance'], label='Matched Radius')
axes[2].plot(filtered_df2['Set Emittance'], filtered_df2['Emittance'], label='2*Matched Radius')
axes[2].plot(filtered_df3['Set Emittance'], filtered_df3['Emittance'], label='3*Matched Radius')
axes[2].plot(filtered_df4['Set Emittance'], filtered_df4['Emittance'], label='4*Matched Radius')
axes[2].plot(filtered_df5['Set Emittance'], filtered_df5['Emittance'], label='5*Matched Radius')
axes[2].plot(filtered_df05['Set Emittance'], filtered_df05['Emittance'], label='0.5*Matched Radius')

# Add labels and titles
axes[0].set_ylabel('Beam Energy')
axes[1].set_ylabel('Beam Spread')
axes[2].set_ylabel('Final Emittance')
axes[2].set_xlabel('Set Emittance')

# Add legends
for ax in axes:
    ax.legend()
    ax.grid(True)

plt.tight_layout()  # Ensures better spacing between plots
plt.savefig('Plots/fixed_emittance_varying_radius.png')
plt.show()

filtered_df_check=df[df['Beam Spread']<=0.01]
print(filtered_df_check[['Set Emittance','Set Radius','Beam Spread']])


# Filter data based on different Set Emittances
filtered_df1 = df[df['Set Emittance'] == 2]
filtered_df2 = df[df['Set Emittance'] == 2.7771]
filtered_df3 = df[df['Set Emittance'] == 8.0704]
filtered_df4 = df[df['Set Emittance'] == 10.6094]
filtered_df5 = df[df['Set Emittance'] == 15.56]
filtered_df05 = df[df['Set Emittance'] == 30]

# Print a subset of data for verification
print(filtered_df1[['Set Radius', 'Set Emittance', 'Beam Spread', 'Beam Energy', 'Emittance']])

# Create 3 subplots stacked vertically
fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

# Plot each variable on its own subplot
axes[0].plot(filtered_df1['Set Radius'], filtered_df1['Beam Energy'], label='Set Emittance = 3')
axes[0].plot(filtered_df2['Set Radius'], filtered_df2['Beam Energy'], label='Set Emittance = 2.7771')
axes[0].plot(filtered_df3['Set Radius'], filtered_df3['Beam Energy'], label='Set Emittance = 8.0704')
axes[0].plot(filtered_df4['Set Radius'], filtered_df4['Beam Energy'], label='Set Emittance = 10.6094')
axes[0].plot(filtered_df5['Set Radius'], filtered_df5['Beam Energy'], label='Set Emittance = 15.56')
axes[0].plot(filtered_df05['Set Radius'], filtered_df05['Beam Energy'], label='Set Emittance = 30')

axes[1].plot(filtered_df1['Set Radius'], filtered_df1['Beam Spread'], label='Set Emittance = 2')
axes[1].plot(filtered_df2['Set Radius'], filtered_df2['Beam Spread'], label='Set Emittance = 2.7771')
axes[1].plot(filtered_df3['Set Radius'], filtered_df3['Beam Spread'], label='Set Emittance = 8.0704')
axes[1].plot(filtered_df4['Set Radius'], filtered_df4['Beam Spread'], label='Set Emittance = 10.6094')
axes[1].plot(filtered_df5['Set Radius'], filtered_df5['Beam Spread'], label='Set Emittance = 15.56')
axes[1].plot(filtered_df05['Set Radius'], filtered_df05['Beam Spread'], label='Set Emittance = 30')

axes[2].plot(filtered_df1['Set Radius'], filtered_df1['Emittance'], label='Set Emittance = 2')
axes[2].plot(filtered_df2['Set Radius'], filtered_df2['Emittance'], label='Set Emittance = 2.7771')
axes[2].plot(filtered_df3['Set Radius'], filtered_df3['Emittance'], label='Set Emittance = 8.0704')
axes[2].plot(filtered_df4['Set Radius'], filtered_df4['Emittance'], label='Set Emittance = 10.6094')
axes[2].plot(filtered_df5['Set Radius'], filtered_df5['Emittance'], label='Set Emittance = 15.56')
axes[2].plot(filtered_df05['Set Radius'], filtered_df05['Emittance'], label='Set Emittance = 30')

# Add labels and titles
axes[0].set_ylabel('Beam Energy')
axes[1].set_ylabel('Beam Spread')
axes[2].set_ylabel('Final Emittance')
axes[2].set_xlabel('Set Radius')

# Add legends
for ax in axes:
    ax.legend()
    ax.grid(True)

# Tight layout to ensure proper spacing
plt.tight_layout()

# Save the plot to a file (finish the save path)
plt.savefig('Plots/parameter_variation_vs_emittance.png')

# Show the plot
plt.show()