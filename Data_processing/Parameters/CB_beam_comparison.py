import subprocess
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# Get the directory where the current script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Change the current working directory to the script directory
os.chdir(script_dir)

def extract_properties_sync(output):
    """
    Extract properties from the output string of the synchrotron_carys3D_sys_input.py script.
    This function assumes properties are on separate lines in a specific format.
    """
    properties = {}
    lines = output.strip().split('\n')
    
    for line in lines:
        try:
            # Adjust based on actual output format
            if "UV/Xray ratio:" in line:
                properties['ratio'] = float(line.split(":")[-1].strip().split()[0])  # Extract the first part after ':'
            elif "Mean Theta:" in line:
                properties['mean_theta'] = float(line.split(":")[-1].strip())
            elif "Critical Energy:" in line:
                properties['crit_energy'] = float(line.split(":")[-1].strip())

        except ValueError:
            print(f"Could not convert the line to float: '{line}'")
    
    return properties


def get_properties_sync(save_num,emittance_num):
    """
    Run synchrotron_carys3D_sys_input.py  with given parameters and return the extracted properties.
    """
    cmd = [
        "python",  r"synchrotron_carys3D_sys_input.py",  # Call the synchrotron_carys3D_sys_input.py script         
        "--run_no", str(save_num),"--emittance",str(emittance_num)     # Save number argument, emittance argument

    ]
    
    try:
        # Run the command and capture the output
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Return the extracted properties from the output
        return extract_properties_sync(result.stdout)

    except subprocess.CalledProcessError as e:
        print(f"Error while processing  save {save_num} :")
        print("Error Output:", e.stderr)  # Print the error message
        print("Standard Output:", e.stdout)  # Also print any output captured before the error
        return None
    
def extract_properties_em(output):
    """
    Extract properties from the output string of the em.py script.
    This function assumes properties are on separate lines in a specific format.
    """
    properties = {}
    lines = output.strip().split('\n')
    
    for line in lines:
        try:
            # Adjust based on actual output format
            if "Geometric Emittance:" in line:
                properties['geometric_emittance'] = float(line.split(":")[-1].strip())
            elif "Energy:" in line:
                properties['beam_energy'] = float(line.split(":")[-1].strip())
                print(line)
            elif "Spread:" in line:
                properties['beam_spread'] = float(line.split(":")[-1].strip())
            elif "Beam Radius:" in line:
                properties['beam_radius'] = float(line.split(":")[-1].strip())  
            
            

        except ValueError:
            print(f"Could not convert the line to float: '{line}'")
    
    return properties


def get_properties_em(data_dir,save_num,species):
    """
    Run em.py with given parameters and return the extracted properties.
    """
    cmd = [
        "python", r'em.py',  # Call the em.py script
        data_dir,           # Directory argument
        str(save_num),      # Save number argument
        str(species)        # Species argument
    ]
    
    try:
        # Run the command and capture the output
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Return the extracted properties from the output
        return extract_properties_em(result.stdout)

    except subprocess.CalledProcessError as e:
        print(f"Error while processing  directory {data_dir} :")
        print("Error Output:", e.stderr)  # Print the error message
        print("Standard Output:", e.stdout)  # Also print any output captured before the error
        return None




run_no=40 #change to run number of interest usually last number in scan -1
data_directory=r"emittance_scan" #change to directory within cluster where scan is
species=2
suffix=np.array(["fix_no_CB","fix_CB"])
sub_folders= [f"emittance-{num}" for num in suffix] #change depending on scan
print("sub folder:",sub_folders)

# Lists to hold the extracted properties


#initial_emittance_list=[]



for index, subfolder in enumerate(sub_folders):
    emittance_list = []
    beam_energy_list=[]
    beam_spread_list=[]
    beam_radius_list=[]
    mean_theta_list=[]
    crit_energy_list=[]
    ratio_list = []

    full_path = os.path.join(data_directory, subfolder)
    print("file path:",full_path)

    data_dir=full_path

    # Check if the CSV file already exists before processing
    output_file1 = f'output_{subfolder}_em.csv'
    output_file2 = f'output_{subfolder}_sync.csv'
    if os.path.exists(output_file1) and os.path.exists(output_file2):
        print(f"File {output_file1} and {output_file2} already exists. Skipping this subfolder...")
        continue  # Skip this subfolder if the CSV file already exists

    for i in range(1,run_no+1):
        print(f"Processing save {i} in directory {subfolder}...")
        properties_em=get_properties_em(data_dir,i,species)
        properties_sync=get_properties_sync(run_no-1,str(suffix[index]))
    
        if properties_sync is not None:
            print(f"Directory {subfolder}: Properties = {properties_sync}")
            #save properties to relevant list
            ratio_list.append(properties_sync.get('ratio', None))
            mean_theta_list.append(properties_sync.get('mean_theta', None))
            crit_energy_list.append(properties_sync.get('crit_energy', None))

        else:
            print(f"Directory {subfolder}: Failed to retrieve sync properties.")
        if properties_em is not None:
            print(f"Directory {subfolder}: Properties =  {properties_em}")
            #save properties to relevant list
            emittance_list.append(properties_em.get('geometric_emittance', None))
            beam_energy_list.append(properties_em.get('beam_energy', None))
            beam_spread_list.append(properties_em.get('beam_spread', None))
            beam_radius_list.append(properties_em.get('beam_radius', None))

        else:
            print(f"Directory {subfolder}: Failed to retrieve  em properties.")


    #ratio_array=np.array(ratio_list)
    emittance_array=np.array(emittance_list)
    #initial_emittance_array=np.array(initial_emittance_list)
    #mean_theta_array=np.array(mean_theta_list)
    #crit_energy_array=np.array(crit_energy_list)
    beam_energy_array=np.array(beam_energy_list)
    beam_spread_array=np.array(beam_spread_list)
    beam_radius_array=np.array(beam_radius_list)
    print(beam_energy_array)
    # Create a DataFrame from the two arrays
    df_em = pd.DataFrame()
    df_em = pd.DataFrame({'Emittance': emittance_array, 'Beam Energy':beam_energy_array, 'Beam Spread':beam_spread_array, 'Beam Radius':beam_radius_array})
    df_sync = pd.DataFrame()
    df_sync = pd.DataFrame({'Uv/X-ray': ratio_list,'Mean Theta':mean_theta_list, 'Critical Energy':crit_energy_list})
    # Save to CSV
    df_em.to_csv(fr'output_{subfolder}_em.csv', index=False)
    df_sync.to_csv(fr'output_{subfolder}_sync.csv', index=False)



# Read the first CSV into DataFrame
df1 = pd.read_csv('output_emittance-fix_no_CB_em.csv')
# Read the second CSV into DataFrame
df2 = pd.read_csv('output_emittance-fix_CB_em.csv')

df3=pd.read_csv('output_sync-fix_no_CB_sync.csv')
df4=pd.read_csv('output_sync-fix_CB_sync.csv')

#em features-------------------------------------------------------------------
# Create a figure and axis with multiple subplots
fig, ax1 = plt.subplots(figsize=(10, 6))

# Create additional axes for the other plots
ax2 = ax1.twinx()  # Create another y-axis that shares the x-axis
ax3 = ax1.twinx()  # Create another y-axis that shares the x-axis
ax4 = ax1.twinx()  # Create a fourth y-axis that shares the x-axis

# Offset the additional axes to prevent overlap
ax3.spines['right'].set_position(('outward', 60))
ax4.spines['right'].set_position(('outward', 120))

# Plot Emittance vs Distance for the first file on ax1 (solid line)
ax1.plot(df1.index/4, df1['Emittance'], color='b', label='Emittance (no CB)', linestyle='-', marker='o')
# Plot Emittance vs Distance for the second file on ax1 (dashed line)
ax1.plot(df2.index/4, df2['Emittance'], color='b', label='Emittance (CB)', linestyle='--', marker='x')

# Plot Beam Energy vs Distance for the first file on ax2 (solid line)
ax2.plot(df1.index/4, df1['Beam Energy'], color='g', label='Beam Energy (no CB)', linestyle='-', marker='x')
# Plot Beam Energy vs Distance for the second file on ax2 (dashed line)
ax2.plot(df2.index/4, df2['Beam Energy'], color='g', label='Beam Energy (CB)', linestyle='--', marker='^')

# Plot Beam Spread vs Distance for the first file on ax3 (solid line)
ax3.plot(df1.index/4, df1['Beam Spread'], color='r', label='Beam Spread (no CB)', linestyle='-', marker='^')
# Plot Beam Spread vs Distance for the second file on ax3 (dashed line)
ax3.plot(df2.index/4, df2['Beam Spread'], color='r', label='Beam Spread (CB)', linestyle='--', marker='s')

# Plot Beam Radius vs Distance for the first file on ax4 (solid line)
ax4.plot(df1.index/4, df1['Beam Radius'], color='m', label='Beam Radius (no CB)', linestyle='-', marker='s')
# Plot Beam Radius vs Distance for the second file on ax4 (dashed line)
ax4.plot(df2.index/4, df2['Beam Radius'], color='m', label='Beam Radius (CB)', linestyle='--', marker='D')

# Set axis labels
ax1.set_xlabel('Distance')
ax1.set_ylabel('Emittance', color='b')
ax2.set_ylabel('Beam Energy (GeV)', color='g')
ax3.set_ylabel('Beam Spread', color='r')
ax4.set_ylabel('Beam Radius', color='m')

# Set tick colors
ax1.tick_params(axis='y', labelcolor='b')
ax2.tick_params(axis='y', labelcolor='g')
ax3.tick_params(axis='y', labelcolor='r')
ax4.tick_params(axis='y', labelcolor='m')

# Add legends to the plot with smaller font size
ax1.legend(loc='upper left', bbox_to_anchor=(0.0, -0.1), borderaxespad=0., ncol=1, fontsize=10)
ax2.legend(loc='upper left', bbox_to_anchor=(0.3, -0.1), borderaxespad=0., ncol=1, fontsize=10)
ax3.legend(loc='upper left', bbox_to_anchor=(0.6, -0.1), borderaxespad=0., ncol=1, fontsize=10)
ax4.legend(loc='upper left', bbox_to_anchor=(0.9, -0.1), borderaxespad=0., ncol=1, fontsize=10)


# Add title and grid
fig.suptitle('Comparison of Emittance, Beam Energy, Beam Spread, and Beam Radius (no CB vs CB)')
fig.tight_layout()  # Adjust layout to avoid overlapping labels

# Show the plot
plt.show()
#Sync features--------------------------------------------------------
# Create a figure and axis with multiple subplots
fig2, ax1 = plt.subplots(figsize=(10, 6))

# Create additional axes for the other plots
ax2 = ax1.twinx()  # Create another y-axis that shares the x-axis
ax3 = ax1.twinx()  # Create another y-axis that shares the x-axis
ax4 = ax1.twinx()  # Create a fourth y-axis that shares the x-axis

# Offset the additional axes to prevent overlap
ax3.spines['right'].set_position(('outward', 60))


# Plot Emittance vs Distance for the first file on ax1 (solid line)
ax1.plot(df1.index, df3['Uv/X-ray'], color='b', label='UV:Xray (no CB)', linestyle='-', marker='o')
# Plot Emittance vs Distance for the second file on ax1 (dashed line)
ax1.plot(df2.index, df4['Uv/X-ray '], color='b', label='UV:Xray  (CB)', linestyle='--', marker='x')

# Plot Beam Energy vs Distance for the first file on ax2 (solid line)
ax2.plot(df1.index, df3['Mean Theta'], color='g', label='Mean Theta (no CB)', linestyle='-', marker='x')
# Plot Beam Energy vs Distance for the second file on ax2 (dashed line)
ax2.plot(df2.index, df4['Mean Theta'], color='g', label='Mean Theta (CB)', linestyle='--', marker='^')

# Plot Beam Spread vs Distance for the first file on ax3 (solid line)
ax3.plot(df1.index, df3['Critical Energy'], color='r', label='Critical Energy (no CB)', linestyle='-', marker='^')
# Plot Beam Spread vs Distance for the second file on ax3 (dashed line)
ax3.plot(df2.index, df4['Critical Energy'], color='r', label='Critical Energy (CB)', linestyle='--', marker='s')

# Set axis labels
ax1.set_xlabel('Distance')
ax1.set_ylabel('X-ray/UV', color='b')
ax2.set_ylabel('Mean Theta', color='g')
ax3.set_ylabel('Critical energy', color='r')


# Set tick colors
ax1.tick_params(axis='y', labelcolor='b')
ax2.tick_params(axis='y', labelcolor='g')
ax3.tick_params(axis='y', labelcolor='r')


# Add legends to the plot with smaller font size
ax1.legend(loc='upper left', bbox_to_anchor=(0.0, -0.1), borderaxespad=0., ncol=1, fontsize=10)
ax2.legend(loc='upper left', bbox_to_anchor=(0.3, -0.1), borderaxespad=0., ncol=1, fontsize=10)
ax3.legend(loc='upper left', bbox_to_anchor=(0.6, -0.1), borderaxespad=0., ncol=1, fontsize=10)



# Add title and grid
fig.suptitle('Comparison of X-ray:UV, mean theta and critical energy (no CB vs CB)')
fig.tight_layout()  # Adjust layout to avoid overlapping labels

# Show the plot
plt.show()


