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




run_no=10 #change to run number of interest usually last number in scan -1
data_directory=r"emittance_scan" #change to directory within cluster where scan is
species=2
# suffix=np.array([str(1), 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 
# 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 
# 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 
# 5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 
# 7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 
# 10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 11.0, 11.1, 11.2, 
# 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9, 12.0, 12.1, 12.2, 12.3, 12.4, 12.5, 
# 12.6, 12.7, 12.8, 12.9, 13.0, 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7, 13.8, 
# 13.9, 14.0, 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7, 14.8, 14.9, 15.0, 15.1, 
# 15.2, 15.3, 15.4, 15.5, 15.6, 15.7, 15.8, 15.9, 16.0])
#suffix=np.array([1.9,2.0])
suffix=np.array([str(1),1.1])
#print(suffix)
sub_folders= [f"emittance-{num}" for num in suffix] #change depending on scan
#print(sub_folders)

# Lists to hold the extracted properties
ratio_list = []
emittance_list = []
initial_emittance_list=[]
mean_theta_list=[]
crit_energy_list=[]
beam_energy_list=[]
beam_spread_list=[]
beam_radius_list=[]


for index, subfolder in enumerate(sub_folders):
    full_path = os.path.join(data_directory, subfolder)

    data_dir=full_path

    properties_sync=get_properties_sync(run_no,str(suffix[index]))
    properties_em=get_properties_em(data_dir,run_no+1,species)
    properties_em_initial=get_properties_em(data_dir,1,species)

    if properties_sync is not None:
        print(f"Directory {subfolder}: Properties = {properties_sync}")
        #save properties to relevant list
        ratio_list.append(properties_sync.get('ratio', None))
        mean_theta_list.append(properties_sync.get('mean_theta', None))
        crit_energy_list.append(properties_sync.get('crit_energy', None))



    else:
        print(f"Directory {subfolder}: Failed to retrieve sync properties.")

    if properties_em is not None:
        print(f"Directory {subfolder}: Properties = final {properties_em}")
        #save properties to relevant list
        emittance_list.append(properties_em.get('geometric_emittance', None))
        beam_energy_list.append(properties_em.get('beam_energy', None))
        beam_spread_list.append(properties_em.get('beam_spread', None))
        beam_radius_list.append(properties_em.get('beam_radius', None))

    else:
        print(f"Directory {subfolder}: Failed to retrieve final em properties.")
    
    if properties_em_initial is not None:
        print(f"Directory {subfolder}: Properties = initial {properties_em_initial}")
        #save properties to relevant list
        initial_emittance_list.append(properties_em_initial.get('geometric_emittance', None))

    else:
        print(f"Directory {subfolder}: Failed to retrieve initial em properties.")


ratio_array=np.array(ratio_list)
emittance_array=np.array(emittance_list)
initial_emittance_array=np.array(initial_emittance_list)
mean_theta_array=np.array(mean_theta_list)
crit_energy_array=np.array(crit_energy_list)
beam_energy_array=np.array(beam_energy_list)
beam_spread_array=np.array(beam_spread_list)
beam_radius_array=np.array(beam_radius_list)
# Create a DataFrame from the two arrays
df = pd.DataFrame({'Emittance': emittance_array, 'Uv/X-ray': ratio_array,'Initial emittance':initial_emittance_array,'Mean Theta':mean_theta_array, 'Critical Energy':crit_energy_array, 'Beam Energy':beam_energy_array, 'Beam Spread':beam_spread_array, 'Beam Radius':beam_radius_array})

# Specify the file path
file_path = 'output.csv'

# Check if the file already exists
if os.path.exists(file_path):
    # If the file exists, append to it (without including the header)
    df.to_csv(file_path, mode='a', header=False, index=False)
else:
    # If the file does not exist, create the file and write the DataFrame with the header
    df.to_csv(file_path, index=False)

