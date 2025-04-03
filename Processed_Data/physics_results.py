import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os


# Get the directory where the current script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Change the current working directory to the script directory
os.chdir(script_dir)

data_set=r"data_sets\big_scan_correct.csv"
df = pd.read_csv(data_set)
#df=df[df['Beam Spread']>0.01]

filtered_df1=df[df['Set Radius']==1]
filtered_df2=df[df['Set Radius']==2]
filtered_df3=df[df['Set Radius']==3]
filtered_df4=df[df['Set Radius']==4]
filtered_df5=df[df['Set Radius']==5]
filtered_df05=df[df['Set Radius']==0.5]
print(filtered_df1[['Set Radius','Set Emittance','Beam Spread', 'Beam Energy','Emittance']])

# Create 3 subplots stacked vertically
fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

# Plot each variable on its own subplot
axes[0].plot(filtered_df1['Set Emittance'], filtered_df1['Beam Energy'], label=r'$\sigma_{ic}$')
axes[0].plot(filtered_df2['Set Emittance'], filtered_df2['Beam Energy'], label=r'$2 \sigma_{ic}$')
axes[0].plot(filtered_df3['Set Emittance'], filtered_df3['Beam Energy'], label=r'$3 \sigma_{ic}$')
axes[0].plot(filtered_df4['Set Emittance'], filtered_df4['Beam Energy'], label=r'$4 \sigma_{ic}$')
axes[0].plot(filtered_df5['Set Emittance'], filtered_df5['Beam Energy'], label=r'$5 \sigma_{ic}$')
axes[0].plot(filtered_df05['Set Emittance'], filtered_df05['Beam Energy'], label=r'$0.5 \sigma_{ic}$')

axes[1].plot(filtered_df1['Set Emittance'], filtered_df1['Beam Spread'], label=r'$\sigma_{ic}$')
axes[1].plot(filtered_df2['Set Emittance'], filtered_df2['Beam Spread'], label=r'$2 \sigma_{ic}$')
axes[1].plot(filtered_df3['Set Emittance'], filtered_df3['Beam Spread'], label=r'$3 \sigma_{ic}$')
axes[1].plot(filtered_df4['Set Emittance'], filtered_df4['Beam Spread'], label=r'$4 \sigma_{ic}$')
axes[1].plot(filtered_df5['Set Emittance'], filtered_df5['Beam Spread'], label=r'$5 \sigma_{ic}$')
axes[1].plot(filtered_df05['Set Emittance'], filtered_df05['Beam Spread'], label=r'$0.5 \sigma_{ic}$')

axes[2].plot(filtered_df1['Set Emittance'], filtered_df1['Emittance'], label=r'$\sigma_{ic}$')
axes[2].plot(filtered_df2['Set Emittance'], filtered_df2['Emittance'], label=r'$2 \sigma_{ic}$')
axes[2].plot(filtered_df3['Set Emittance'], filtered_df3['Emittance'], label=r'$3 \sigma_{ic}$')
axes[2].plot(filtered_df4['Set Emittance'], filtered_df4['Emittance'], label=r'$4 \sigma_{ic}$')
axes[2].plot(filtered_df5['Set Emittance'], filtered_df5['Emittance'], label=r'$5 \sigma_{ic}$')
axes[2].plot(filtered_df05['Set Emittance'], filtered_df05['Emittance'], label=r'$0.5 \sigma_{ic}$')

# Add labels and titles
axes[0].set_ylabel('Beam Energy (GeV)')
axes[1].set_ylabel('Beam Energy Spread (%)')
axes[2].set_ylabel(r'Final Emittance ($\mu m$)')
axes[2].set_xlabel(r'Set Emittance ($\mu m$)')

# Add legends
for ax in axes:
    ax.legend(ncol=2, fontsize=10, title='Set radius')
    ax.grid(True)

plt.suptitle("Final beam parameters against set emittance and radius", fontsize=18)
plt.savefig('Plots/fixed_emittance_varying_radius.png')
plt.show()

filtered_df_check=df[df['Beam Spread']<=0.01]
print(filtered_df_check[['Set Emittance','Set Radius','Beam Spread']])


# Filter data based on different Set Emittances
filtered_df1 = df[df['Set Emittance'] == 2]
filtered_df2 = df[df['Set Emittance'] == 4.5455]
filtered_df3 = df[df['Set Emittance'] == 8.5051]
filtered_df4 = df[df['Set Emittance'] == 13.3131]
filtered_df5 = df[df['Set Emittance'] == 20.3838]
filtered_df05 = df[df['Set Emittance'] == 30]

# Print a subset of data for verification
print(filtered_df1[['Set Radius', 'Set Emittance', 'Beam Spread', 'Beam Energy', 'Emittance']])

# Create 3 subplots stacked vertically
fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

# Plot each variable on its own subplot
axes[0].plot(filtered_df1['Set Radius'], filtered_df1['Beam Energy'], label=r'2$\mu m$')
axes[0].plot(filtered_df2['Set Radius'], filtered_df2['Beam Energy'], label=r'4.5$\mu m$')
axes[0].plot(filtered_df3['Set Radius'], filtered_df3['Beam Energy'], label=r'8.5$\mu m$')
axes[0].plot(filtered_df4['Set Radius'], filtered_df4['Beam Energy'], label=r'13.3$\mu m$')
axes[0].plot(filtered_df5['Set Radius'], filtered_df5['Beam Energy'], label=r'20.4$\mu m$')
axes[0].plot(filtered_df05['Set Radius'], filtered_df05['Beam Energy'], label=r'30$\mu m$')

axes[1].plot(filtered_df1['Set Radius'], filtered_df1['Beam Spread'], label=r'2$\mu m$')
axes[1].plot(filtered_df2['Set Radius'], filtered_df2['Beam Spread'], label=r'4.5$\mu m$')
axes[1].plot(filtered_df3['Set Radius'], filtered_df3['Beam Spread'], label=r'8.5$\mu m$')
axes[1].plot(filtered_df4['Set Radius'], filtered_df4['Beam Spread'], label=r'13.3$\mu m$')
axes[1].plot(filtered_df5['Set Radius'], filtered_df5['Beam Spread'], label=r'20.4$\mu m$')
axes[1].plot(filtered_df05['Set Radius'], filtered_df05['Beam Spread'], label=r'30$\mu m$')

axes[2].plot(filtered_df1['Set Radius'], filtered_df1['Emittance'], label=r'2$\mu m$')
axes[2].plot(filtered_df2['Set Radius'], filtered_df2['Emittance'], label=r'4.5$\mu m$')
axes[2].plot(filtered_df3['Set Radius'], filtered_df3['Emittance'], label=r'8.5$\mu m$')
axes[2].plot(filtered_df4['Set Radius'], filtered_df4['Emittance'], label=r'13.3$\mu m$')
axes[2].plot(filtered_df5['Set Radius'], filtered_df5['Emittance'], label=r'20.4$\mu m$')
axes[2].plot(filtered_df05['Set Radius'], filtered_df05['Emittance'], label=r'30$\mu m$')

# Add labels and titles
axes[0].set_ylabel('Beam Energy (GeV)')
axes[1].set_ylabel('Beam Energy Spread (%)')
axes[2].set_ylabel(r'Final Emittance ($\mu m$)')
axes[2].set_xlabel(r'Set Radius ($\sigma_{ic}$)')

# Add legends
for ax in axes:
    ax.legend(ncol=2, fontsize=10, title='Set emittance')
    ax.grid(True)

# Tight layout to ensure proper spacing
#plt.tight_layout()

# Save the plot to a file (finish the save path)
plt.suptitle("Final beam parameters against set emittance and radius", fontsize=16)
plt.savefig('Plots/parameter_variation_vs_emittance.png')

# Show the plot
plt.show()