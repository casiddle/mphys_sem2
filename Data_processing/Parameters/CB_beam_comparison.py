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




run_no=11 #change to run number of interest usually last number in scan -1
data_directory=r"emittance_scan" #change to directory within cluster where scan is
species=2
suffix=np.array(["no_CB","CB"])
sub_folders= [f"emittance-{num}" for num in suffix] #change depending on scan
#print(sub_folders)

# Lists to hold the extracted properties
#ratio_list = []

#initial_emittance_list=[]
#mean_theta_list=[]
#crit_energy_list=[]



for index, subfolder in enumerate(sub_folders):
    emittance_list = []
    beam_energy_list=[]
    beam_spread_list=[]
    beam_radius_list=[]
    full_path = os.path.join(data_directory, subfolder)

    data_dir=full_path

    # Check if the CSV file already exists before processing
    output_file = f'output_{subfolder}.csv'
    if os.path.exists(output_file):
        print(f"File {output_file} already exists. Skipping this subfolder...")
        continue  # Skip this subfolder if the CSV file already exists

    for i in range(1,run_no+1):
        print(f"Processing save {i} in directory {subfolder}...")
        properties_em=get_properties_em(data_dir,i,species)
        #properties_sync=get_properties_sync(run_no-1,str(suffix[index]))

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
    # Create a DataFrame from the two arrays
    df = pd.DataFrame()
    df = pd.DataFrame({'Emittance': emittance_array, 'Beam Energy':beam_energy_array, 'Beam Spread':beam_spread_array, 'Beam Radius':beam_radius_array})

    # Save to CSV
    df.to_csv(fr'output_{subfolder}.csv', index=False)



# Read the first CSV into DataFrame
df1 = pd.read_csv('output_emittance-no_CB.csv')

# Read the second CSV into DataFrame
df2 = pd.read_csv('output_emittance-CB.csv')

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
ax1.plot(df1.index, df1['Emittance'], color='b', label='Emittance (no CB)', linestyle='-', marker='o')
# Plot Emittance vs Distance for the second file on ax1 (dashed line)
ax1.plot(df2.index, df2['Emittance'], color='b', label='Emittance (CB)', linestyle='--', marker='x')

# Plot Beam Energy vs Distance for the first file on ax2 (solid line)
ax2.plot(df1.index, df1['Beam Energy'], color='g', label='Beam Energy (no CB)', linestyle='-', marker='x')
# Plot Beam Energy vs Distance for the second file on ax2 (dashed line)
ax2.plot(df2.index, df2['Beam Energy'], color='g', label='Beam Energy (CB)', linestyle='--', marker='^')

# Plot Beam Spread vs Distance for the first file on ax3 (solid line)
ax3.plot(df1.index, df1['Beam Spread'], color='r', label='Beam Spread (no CB)', linestyle='-', marker='^')
# Plot Beam Spread vs Distance for the second file on ax3 (dashed line)
ax3.plot(df2.index, df2['Beam Spread'], color='r', label='Beam Spread (CB)', linestyle='--', marker='s')

# Plot Beam Radius vs Distance for the first file on ax4 (solid line)
ax4.plot(df1.index, df1['Beam Radius'], color='m', label='Beam Radius (no CB)', linestyle='-', marker='s')
# Plot Beam Radius vs Distance for the second file on ax4 (dashed line)
ax4.plot(df2.index, df2['Beam Radius'], color='m', label='Beam Radius (CB)', linestyle='--', marker='D')

# Set axis labels
ax1.set_xlabel('Distance')
ax1.set_ylabel('Emittance', color='b')
ax2.set_ylabel('Beam Energy', color='g')
ax3.set_ylabel('Beam Spread', color='r')
ax4.set_ylabel('Beam Radius', color='m')

# Set tick colors
ax1.tick_params(axis='y', labelcolor='b')
ax2.tick_params(axis='y', labelcolor='g')
ax3.tick_params(axis='y', labelcolor='r')
ax4.tick_params(axis='y', labelcolor='m')

ax1.legend(loc='upper left')
ax2.legend(loc='upper right') 
ax3.legend(loc='lower left')
ax4.legend(loc='lower right')

# Add title and grid
fig.suptitle('Comparison of Emittance, Beam Energy, Beam Spread, and Beam Radius (Files 1 vs 2)')
fig.tight_layout()  # Adjust layout to avoid overlapping labels

# Show the plot
plt.show()

