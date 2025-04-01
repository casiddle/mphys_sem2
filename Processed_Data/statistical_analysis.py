import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize
import os
import dcor 
from sklearn.feature_selection import mutual_info_regression

# Get the directory where the current script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Change the current working directory to the script directory
os.chdir(script_dir)

def theta_to_r(theta, distance):
    r=distance*np.tan(theta)*1000
    return r

def distance_correlation_calculator(df, targets):
    correlation_list = []

    for target in targets:
        dist_corr = {
            col: dcor.distance_correlation(df[col], df[target]) 
            if df[col].nunique() > 1 else np.nan 
            for col in df.columns
        }

        # Append results directly to the list with target included
        correlation_list.extend([(target, feature, corr) for feature, corr in dist_corr.items()])

    # Create DataFrame for sorted correlation list
    correlation_df = pd.DataFrame(correlation_list, columns=['Target', 'Feature', 'Correlation'])
    correlation_df_sorted = correlation_df.sort_values(by='Correlation', ascending=False)

    # Create reshaped DataFrame for heatmap
    reshaped_df = correlation_df.pivot(index='Feature', columns='Target', values='Correlation')

    return reshaped_df, correlation_df_sorted


data_set=r"data_sets\big_scan_correct.csv"
df = pd.read_csv(data_set)
#df = df.drop(columns=['Initial emittance'])


# Apply the function to the column
df['X-ray Mean Radiation Radius'] = df['X-ray Mean Theta'].apply(lambda theta: theta_to_r(theta, 11))
df['UV Percentage']=df['No. UV Photons']/df['Total no. Photons']
df['Other Percentage']=df['No. Other Photons']/df['Total no. Photons']
df['Total Percentage']=df['UV Percentage']+df['X-ray Percentage']+df['Other Percentage']
#print(df.head())
df = df.drop(columns=['No. UV Photons','No. Other Photons','No. X-ray Photons','Total Percentage'])

#print(df.columns)
print(df.describe(include='all'))

print("RANGES")
print("Emittance ranged from ",df['Emittance'].min()," to ",df['Emittance'].max())
print("Beam Spread ranged from ",df['Beam Spread'].min()," to ",df['Beam Spread'].max())
print("Beam Energy ranged from ",df['Beam Energy'].min()," to ",df['Beam Energy'].max())

#Calculating distance correlation using distance correlation
# List of multiple target variables
targets = ['Emittance', 'Beam Spread', 'Beam Energy']

reshaped_df,ranked_correlation=distance_correlation_calculator(df,targets)
ranked_correlation.to_csv('correlation_matrix_ranked.csv')

#print(ranked_correlation)

plt.figure(figsize=(12, 8))
sns.heatmap(reshaped_df, annot=True, cmap='rocket_r', fmt=".2f", cbar=True)
plt.title("Distance Correlation Analysis Heatmap")
plt.savefig('Plots/distance_correlation_heatmap.png')
#plt.show()

#___________________________________________________________________________________________________________________________




# Filter rows where 'Feature 1' or 'Feature 2' contains 'Emittance'
emittance_filtered_df = ranked_correlation[(ranked_correlation["Target"]=="Emittance")]
#print(emittance_filtered_df.head(10))

beam_spread_filtered_df = ranked_correlation[(ranked_correlation["Target"]=="Beam Spread")]
#print(beam_spread_filtered_df.head(10))

beam_energy_filtered_df = ranked_correlation[(ranked_correlation["Target"]=="Beam Energy")]
#print(beam_energy_filtered_df.head(10))


X = df[['Emittance','Beam Spread','Beam Energy', 'X-ray Percentage',
        'X-ray Critical Energy','X-ray Mean Radiation Radius']]

matrix_df,rabked_correlation=distance_correlation_calculator(X,targets)
#print(matrix_df.head(10))
matrix_df=matrix_df.drop(matrix_df[matrix_df.index.isin(targets)].index)
fig, ax = plt.subplots(figsize=(12, 8))  # Adjust figsize
sns.heatmap(matrix_df, annot=True, cmap='rocket_r', fmt=".2f", cbar=True, 
            annot_kws={'size': 22, 'weight': 'bold'}, ax=ax,
            xticklabels=True, yticklabels=True)
# Adjust the aspect ratio
ax.set_aspect('auto')  # Or you can set a specific aspect ratio like 0.8 or 1.5
plt.title("Distance Correlation Analysis Heatmap",fontsize=20, fontweight='bold')
# Increase font size for x and y axis labels (tick labels)
plt.tick_params(axis='x', labelsize=16)  # X-axis tick labels
plt.tick_params(axis='y', labelsize=16,rotation=30)  # Y-axis tick labels
plt.tight_layout()
plt.savefig('Plots/distance_correlation_heatmap_reduced_parameters.png')
#plt.show()



set_radius=1
emittance_check_df=df[['Emittance','X-ray Percentage',
        'X-ray Critical Energy','X-ray Mean Radiation Radius','UV Percentage','Other Percentage','Critical Energy','Set Radius','Initial emittance']]

red_points = emittance_check_df[emittance_check_df['Set Radius'] ==1]
other_points = emittance_check_df[emittance_check_df['Set Radius'] ==5]
points_05 = emittance_check_df[emittance_check_df['Set Radius'] == 0.5]
points_2 = emittance_check_df[emittance_check_df['Set Radius'] == 2]
points_3 = emittance_check_df[emittance_check_df['Set Radius'] == 3]
points_4 = emittance_check_df[emittance_check_df['Set Radius'] == 4]



# Create a figure with a 2x3 grid of subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 12))

# Plot Emittance vs Mean Radiation Radius on the first subplot (axes[0, 0])
axes[0].scatter(red_points['Emittance'], red_points['X-ray Mean Radiation Radius'], color='red', alpha=0.6, label=f'Set Radius = 1')
axes[0].scatter(other_points['Emittance'], other_points['X-ray Mean Radiation Radius'], color='blue', alpha=0.6, label=f'Set Radius = 5')
axes[0].scatter(red_points['Emittance'], points_05['X-ray Mean Radiation Radius'], color='green', alpha=0.6, label=f'Set Radius = 0.5')
axes[0].scatter(red_points['Emittance'], points_2['X-ray Mean Radiation Radius'], color='magenta', alpha=0.6, label=f'Set Radius = 2')
axes[0].scatter(red_points['Emittance'], points_3['X-ray Mean Radiation Radius'], color='orange', alpha=0.6, label=f'Set Radius = 3')
axes[0].scatter(red_points['Emittance'], points_4['X-ray Mean Radiation Radius'], color='black', alpha=0.6, label=f'Set Radius = 4')
axes[0].set_title('Emittance vs X-ray Mean Radiation Radius')
axes[0].set_xlabel('Emittance')
axes[0].set_ylabel('X-ray Mean Radiation Radius')
axes[0].grid(True)

# Plot Emittance vs Critical Energy on the second subplot (axes[0, 1])
axes[1].scatter(red_points['Emittance'], red_points['X-ray Critical Energy'], color='red', alpha=0.6, label=f'Set Radius = 1')
axes[1].scatter(other_points['Emittance'], other_points['X-ray Critical Energy'], color='blue', alpha=0.6, label=f'Set Radius = 5')
axes[1].scatter(red_points['Emittance'], points_05['X-ray Critical Energy'], color='green', alpha=0.6, label=f'Set Radius = 0.5')
axes[1].scatter(red_points['Emittance'], points_2['X-ray Critical Energy'], color='magenta', alpha=0.6, label=f'Set Radius = 2')
axes[1].scatter(red_points['Emittance'], points_3['X-ray Critical Energy'], color='orange', alpha=0.6, label=f'Set Radius = 3')
axes[1].scatter(red_points['Emittance'], points_4['X-ray Critical Energy'], color='black', alpha=0.6, label=f'Set Radius = 4')
axes[1].set_title('Emittance vs X-ray Critical Energy')
axes[1].set_xlabel('Emittance')
axes[1].set_ylabel('X-ray Critical Energy')
axes[1].grid(True)


# Plot Emittance vs X-ray Percentage on the fourth subplot (axes[1, 0])
axes[2].scatter(red_points['Emittance'], red_points['X-ray Percentage'], color='red', alpha=0.6, label=f'Set Radius = 1')
axes[2].scatter(other_points['Emittance'], other_points['X-ray Percentage'], color='blue', alpha=0.6, label=f'Set Radius = 5')
axes[2].scatter(red_points['Emittance'], points_05['X-ray Percentage'], color='green', alpha=0.6, label=f'Set Radius = 0.5')
axes[2].scatter(red_points['Emittance'], points_2['X-ray Percentage'], color='magenta', alpha=0.6, label=f'Set Radius = 2')
axes[2].scatter(red_points['Emittance'], points_3['X-ray Percentage'], color='orange', alpha=0.6, label=f'Set Radius = 3')
axes[2].scatter(red_points['Emittance'], points_4['X-ray Percentage'], color='black', alpha=0.6, label=f'Set Radius = 4')
axes[2].set_title('Emittance vs X-ray Percentage')
axes[2].set_xlabel('Emittance')
axes[2].set_ylabel('X-ray Percentage')
axes[2].grid(True)




# Adjust layout for better spacing
plt.tight_layout()
# Show the plots
plt.legend()
plt.savefig('Plots/emittance_parameters_plot.png')
#plt.show()

# Create the DataFrame with relevant columns
beam_spread_check_df = df[['Beam Spread', 'X-ray Percentage', 'X-ray Critical Energy', 'X-ray Mean Radiation Radius', 
                           'UV Percentage', 'Other Percentage', 'Critical Energy', 'Set Radius','Initial emittance']]

# Split the data based on Set Radius
red_points = beam_spread_check_df[beam_spread_check_df['Set Radius'] ==1]
other_points = beam_spread_check_df[beam_spread_check_df['Set Radius'] ==5]
points_05 = beam_spread_check_df[beam_spread_check_df['Set Radius'] == 0.5]
points_2 = beam_spread_check_df[beam_spread_check_df['Set Radius'] == 2]
points_3 = beam_spread_check_df[beam_spread_check_df['Set Radius'] == 3]
points_4 = beam_spread_check_df[beam_spread_check_df['Set Radius'] == 4]


# Create a figure with a 2x3 grid of subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 12))

# Plot Emittance vs Mean Radiation Radius on the first subplot (axes[0, 0])
axes[0].scatter(red_points['Beam Spread'], red_points['X-ray Mean Radiation Radius'], color='red', alpha=0.6, label=f'Set Radius = 1')
axes[0].scatter(other_points['Beam Spread'], other_points['X-ray Mean Radiation Radius'], color='blue', alpha=0.6, label=f'Set Radius = 5')
axes[0].scatter(red_points['Beam Spread'], points_05['X-ray Mean Radiation Radius'], color='green', alpha=0.6, label=f'Set Radius = 0.5')
axes[0].scatter(red_points['Beam Spread'], points_2['X-ray Mean Radiation Radius'], color='magenta', alpha=0.6, label=f'Set Radius = 2')
axes[0].scatter(red_points['Beam Spread'], points_3['X-ray Mean Radiation Radius'], color='orange', alpha=0.6, label=f'Set Radius = 3')
axes[0].scatter(red_points['Beam Spread'], points_4['X-ray Mean Radiation Radius'], color='black', alpha=0.6, label=f'Set Radius = 4')
axes[0].set_title('Beam Spread vs X-ray Mean Radiation Radius')
axes[0].set_xlabel('Beam Spread')
axes[0].set_ylabel('X-ray Mean Radiation Radius')
axes[0].grid(True)

# Plot Emittance vs Critical Energy on the second subplot (axes[0, 1])
axes[1].scatter(red_points['Beam Spread'], red_points['X-ray Critical Energy'], color='red', alpha=0.6, label=f'Set Radius = 1')
axes[1].scatter(other_points['Beam Spread'], other_points['X-ray Critical Energy'], color='blue', alpha=0.6, label=f'Set Radius = 5')
axes[1].scatter(red_points['Beam Spread'], points_05['X-ray Critical Energy'], color='green', alpha=0.6, label=f'Set Radius = 0.5')
axes[1].scatter(red_points['Beam Spread'], points_2['X-ray Critical Energy'], color='magenta', alpha=0.6, label=f'Set Radius = 2')
axes[1].scatter(red_points['Beam Spread'], points_3['X-ray Critical Energy'], color='orange', alpha=0.6, label=f'Set Radius = 3')
axes[1].scatter(red_points['Beam Spread'], points_4['X-ray Critical Energy'], color='black', alpha=0.6, label=f'Set Radius = 4')
axes[1].set_title('Beam Spread vs X-ray Critical Energy')
axes[1].set_xlabel('Beam Spread')
axes[1].set_ylabel('X-ray Critical Energy')
axes[1].grid(True)


# Plot Emittance vs X-ray Percentage on the fourth subplot (axes[1, 0])
axes[2].scatter(red_points['Beam Spread'], red_points['X-ray Percentage'], color='red', alpha=0.6, label=f'Set Radius = 1')
axes[2].scatter(other_points['Beam Spread'], other_points['X-ray Percentage'], color='blue', alpha=0.6, label=f'Set Radius = 5')
axes[2].scatter(red_points['Beam Spread'], points_05['X-ray Percentage'], color='green', alpha=0.6, label=f'Set Radius = 0.5')
axes[2].scatter(red_points['Beam Spread'], points_2['X-ray Percentage'], color='magenta', alpha=0.6, label=f'Set Radius = 2')
axes[2].scatter(red_points['Beam Spread'], points_3['X-ray Percentage'], color='orange', alpha=0.6, label=f'Set Radius = 3')
axes[2].scatter(red_points['Beam Spread'], points_4['X-ray Percentage'], color='black', alpha=0.6, label=f'Set Radius = 4')
axes[2].set_title('Beam Spread vs X-ray Percentage')
axes[2].set_xlabel('Beam Spread')
axes[2].set_ylabel('X-ray Percentage')
axes[2].grid(True)


# Adjust layout for better spacing
plt.tight_layout()

# Show the plots
plt.legend()
plt.savefig('Plots/beam_spread_parameters_plot.png')
#plt.show()

# Create the DataFrame with relevant columns
beam_energy_check_df = df[['Beam Energy', 'X-ray Percentage', 'X-ray Critical Energy', 'X-ray Mean Radiation Radius', 
                           'UV Percentage', 'Other Percentage', 'Critical Energy', 'Set Radius','Initial emittance']]

# Split the data based on Set Radius
red_points = beam_energy_check_df[beam_energy_check_df['Set Radius'] ==1]
other_points = beam_energy_check_df[beam_energy_check_df['Set Radius'] ==5]
points_05 = beam_energy_check_df[beam_energy_check_df['Set Radius'] == 0.5]
points_2 = beam_energy_check_df[beam_energy_check_df['Set Radius'] == 2]
points_3 = beam_energy_check_df[beam_energy_check_df['Set Radius'] == 3]
points_4 = beam_energy_check_df[beam_energy_check_df['Set Radius'] == 4]

# Create a figure with a 2x3 grid of subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 12))

# Plot Emittance vs Mean Radiation Radius on the first subplot (axes[0, 0])
axes[0].scatter(red_points['Beam Energy'], red_points['X-ray Mean Radiation Radius'], color='red', alpha=0.6, label=f'Set Radius = 1')
axes[0].scatter(other_points['Beam Energy'], other_points['X-ray Mean Radiation Radius'], color='blue', alpha=0.6, label=f'Set Radius = 5')
axes[0].scatter(red_points['Beam Energy'], points_05['X-ray Mean Radiation Radius'], color='green', alpha=0.6, label=f'Set Radius = 0.5')
axes[0].scatter(red_points['Beam Energy'], points_2['X-ray Mean Radiation Radius'], color='magenta', alpha=0.6, label=f'Set Radius = 2')
axes[0].scatter(red_points['Beam Energy'], points_3['X-ray Mean Radiation Radius'], color='orange', alpha=0.6, label=f'Set Radius = 3')
axes[0].scatter(red_points['Beam Energy'], points_4['X-ray Mean Radiation Radius'], color='black', alpha=0.6, label=f'Set Radius = 4')
axes[0].set_title('Beam Energy vs X-ray Mean Radiation Radius')
axes[0].set_xlabel('Beam Energy')
axes[0].set_ylabel('X-ray Mean Radiation Radius')
axes[0].grid(True)

# Plot Emittance vs Critical Energy on the second subplot (axes[0, 1])
axes[1].scatter(red_points['Beam Energy'], red_points['X-ray Critical Energy'], color='red', alpha=0.6, label=f'Set Radius = 1')
axes[1].scatter(other_points['Beam Energy'], other_points['X-ray Critical Energy'], color='blue', alpha=0.6, label=f'Set Radius = 5')
axes[1].scatter(red_points['Beam Energy'], points_05['X-ray Critical Energy'], color='green', alpha=0.6, label=f'Set Radius = 0.5')
axes[1].scatter(red_points['Beam Energy'], points_2['X-ray Critical Energy'], color='magenta', alpha=0.6, label=f'Set Radius = 2')
axes[1].scatter(red_points['Beam Energy'], points_3['X-ray Critical Energy'], color='orange', alpha=0.6, label=f'Set Radius = 3')
axes[1].scatter(red_points['Beam Energy'], points_4['X-ray Critical Energy'], color='black', alpha=0.6, label=f'Set Radius = 4')
axes[1].set_title('Beam Energy vs X-ray Critical Energy')
axes[1].set_xlabel('Beam Energy')
axes[1].set_ylabel('X-ray Critical Energy')
axes[1].grid(True)


# Plot Emittance vs X-ray Percentage on the fourth subplot (axes[1, 0])
axes[2].scatter(red_points['Beam Energy'], red_points['X-ray Percentage'], color='red', alpha=0.6, label=f'Set Radius = 1')
axes[2].scatter(other_points['Beam Energy'], other_points['X-ray Percentage'], color='blue', alpha=0.6, label=f'Set Radius = 5')
axes[2].scatter(red_points['Beam Energy'], points_05['X-ray Percentage'], color='green', alpha=0.6, label=f'Set Radius = 0.5')
axes[2].scatter(red_points['Beam Energy'], points_2['X-ray Percentage'], color='magenta', alpha=0.6, label=f'Set Radius = 2')
axes[2].scatter(red_points['Beam Energy'], points_3['X-ray Percentage'], color='orange', alpha=0.6, label=f'Set Radius = 3')
axes[2].scatter(red_points['Beam Energy'], points_4['X-ray Percentage'], color='black', alpha=0.6, label=f'Set Radius = 4')
axes[2].set_title('Beam Energy vs X-ray Percentage')
axes[2].set_xlabel('Beam Energy')
axes[2].set_ylabel('X-ray Percentage')
axes[2].grid(True)

# Adjust layout for better spacing
plt.tight_layout()

# Show the plots
plt.legend()
plt.savefig('Plots/beam_energy_parameters_plot.png')
plt.show()

###########################################################################

