import subprocess
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import re

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
            if "Xray/UV ratio:" in line:
                properties['ratio'] = float(line.split(":")[-1].strip().split()[0])  # Extract the first part after ':'
            elif "Mean Theta:" in line:
                properties['mean_theta'] = float(line.split(":")[-1].strip())
            elif "Critical Energy:" in line:
                properties['crit_energy'] = float(line.split(":")[-1].strip())
            elif "X-ray percentage:" in line:
                properties['x_ray_percentage'] = float(line.split(":")[-1].strip())
            elif "Critical Energy X-ray photons:" in line:
                properties['x_ray_crit_energy'] = float(line.split(":")[-1].strip())
            elif "No. X-ray photons:" in line:
                properties['no_x_ray_photons'] = float(line.split(":")[-1].strip())
            elif "No. UV photons:" in line:
                properties['no_uv_photons'] = float(line.split(":")[-1].strip())
            elif "No. Other photons:" in line:
                properties['no_other_photons'] = float(line.split(":")[-1].strip())
            elif "Total no. photons:" in line:
                properties['total_no_photons'] = float(line.split(":")[-1].strip())
            
            
            
            

        except ValueError:
            print(f"Could not convert the line to float: '{line}'")
    
    return properties


def get_properties_sync(save_num,emittance_num,radius_frac):
    """
    Run synchrotron_carys3D_sys_input.py  with given parameters and return the extracted properties.
    """
    cmd = [
        "python",  r"synchrotron_carys3D_sys_input.py",  # Call the synchrotron_carys3D_sys_input.py script         
        "--run_no", str(save_num),"--emittance",str(emittance_num), "--radius_frac",str(radius_frac)    # Save number argument, emittance argument

    ]
    
    try:
        # Run the command and capture the output
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        
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
            elif "Total Charge:" in line:
                properties['total_charge'] = float(line.split(":")[-1].strip())   
            
            

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


def find_largest_file_number(parent_dir):
    """
    Given a parent directory path, this function finds the largest number
    at the end of filenames that follow the pattern 'filename_XXXXX.h5', 
    where 'XXXXX' is a number (e.g., v3d_synchrotron_00001.h5, v3d_synchrotron_00011.h5, ...).
    
    :param parent_dir: str, path to the parent directory where the files are located.
    :return: int, the largest number found in the filenames.
    """
    # Get the list of all files in the directory
    files = os.listdir(parent_dir)

    # Initialize a variable to keep track of the largest number
    largest_number = -1

    # Regex pattern to match the numerical part just before '.h5' (e.g., _00001)
    pattern = r'v3d_synchrotron_(\d+)\.h5$'

    # Iterate over the files
    for file in files:
        # Try to match the pattern at the end of the filename
        match = re.search(pattern, file)
        if match:
            # Extract the number from the match
            number = int(match.group(1))
            
            # Update largest_number if this one is larger
            if number > largest_number:
                largest_number = number

    # Return the largest number found, or -1 if no numbers were found
    return largest_number

# Function to extract numbers from the folder names
def extract_numbers_from_subfolder(subfolder):
    match = re.match(r"emittance-(\d+\.?\d*)_radius-(\d+)", subfolder)
    if match:
        emittance = float(match.group(1))  # Extract the emittance number (convert to float if needed)
        radius = int(match.group(2))       # Extract the radius number (convert to int)
        return emittance, radius
    else:
        return None, None  # Return None if no match found


run_no=11 #change to run number of interest usually last number in scan 
data_directory=r"emittance_scan" #change to directory within cluster where scan is
species=2 #witness beam

suffix_file=r'..\..\Simulations\Beam_builder\test_array.csv'

# Initialize lists for each column
emittance = []
beam_radius = []
beam_radius_fraction = []

# Read the CSV file manually
with open(suffix_file, "r") as file:
    lines = file.readlines()[1:]  # Skip the header row

    for line in lines:
        print("line:", line)
        values = line.strip().split(",")  # Split by comma
        emittance.append(float(values[0]))
        beam_radius.append(float(values[1]))
        beam_radius_fraction.append(float(values[2]))

# Output the lists
print("Emittance:", emittance)
print("Beam Radius:", beam_radius)
print("Beam Radius Fraction:", beam_radius_fraction)


suffix1=emittance #np.array([1.0,1.1])
suffix2=beam_radius_fraction#np.array([1.0])
print("suffix 1:",suffix1)
print("suffix 2:",suffix2)

# Initialize an empty list to store all the subfolders # code for when we are chnaging both radius and emittance
sub_folders = []

# Generate the subfolder names
for num,rad in zip(suffix1, suffix2):
    sub_folders.append(f"emittance-{num}_radius-{rad}")

# Print the result
print(sub_folders)
#check=input("press enter")
# Lists to hold the extracted properties
ratio_list = []
emittance_list = []
initial_emittance_list=[]
mean_theta_list=[]
crit_energy_list=[]
beam_energy_list=[]
beam_spread_list=[]
beam_radius_list=[]
x_ray_percentage_list=[]
x_ray_crit_energy_list=[]
set_emittance_list=[]
set_radius_list=[]
no_x_ray_photons_list=[]
no_uv_photons_list=[]
no_other_photons_list=[]
total_no_photons_list=[]
total_charge_list=[]


for index, subfolder in enumerate(sub_folders):
    full_path = os.path.join(data_directory, subfolder)
    run_no=find_largest_file_number(full_path)
    em,rad=extract_numbers_from_subfolder(subfolder)
    print("EM:",em)
    print("RAD:",rad)
    set_emittance_list.append(em)
    set_radius_list.append(rad)
    #print("Largest file number:",run_no)

    data_dir=full_path

    #properties_sync=get_properties_sync(run_no,str(suffix1[index]),1)
    properties_sync=get_properties_sync(run_no,str(em),str(rad))
    properties_em=get_properties_em(data_dir,run_no,species)
    properties_em_initial=get_properties_em(data_dir,1,species)

    if properties_sync is not None:
        print(f"Directory {subfolder}: Properties = {properties_sync}")
        #save properties to relevant list
        ratio_list.append(properties_sync.get('ratio', None))
        mean_theta_list.append(properties_sync.get('mean_theta', None))
        crit_energy_list.append(properties_sync.get('crit_energy', None))
        x_ray_percentage_list.append(properties_sync.get('x_ray_percentage', None))
        x_ray_crit_energy_list.append(properties_sync.get('x_ray_crit_energy', None))
        no_x_ray_photons_list.append(properties_sync.get('no_x_ray_photons', None))
        no_uv_photons_list.append(properties_sync.get('no_uv_photons', None))
        no_other_photons_list.append(properties_sync.get('no_other_photons', None))
        total_no_photons_list.append(properties_sync.get('total_no_photons', None))



    else:
        print(f"Directory {subfolder}: Failed to retrieve sync properties.")

    if properties_em is not None:
        print(f"Directory {subfolder}: Properties = final {properties_em}")
        #save properties to relevant list
        emittance_list.append(properties_em.get('geometric_emittance', None))
        beam_energy_list.append(properties_em.get('beam_energy', None))
        beam_spread_list.append(properties_em.get('beam_spread', None))
        beam_radius_list.append(properties_em.get('beam_radius', None))
        total_charge_list.append(properties_em.get('total_charge', None))

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
x_ray_percentage_array=np.array(x_ray_percentage_list)
x_ray_crit_energy_array=np.array(x_ray_crit_energy_list)
no_x_ray_photons_array=np.array(no_x_ray_photons_list)
no_uv_photons_array=np.array(no_uv_photons_list)
no_other_photons_array=np.array(no_other_photons_list)
total_no_photons_array=np.array(total_no_photons_list)
total_charge_array=np.array(total_charge_list)

# Create a DataFrame from the two arrays
df = pd.DataFrame({'Emittance': emittance_array, 'X-ray/UV': ratio_array,'Initial emittance':initial_emittance_array,
                   'Mean Theta':mean_theta_array, 'Critical Energy':crit_energy_array, 'Beam Energy':beam_energy_array, 
                   'Beam Spread':beam_spread_array, 'Beam Radius':beam_radius_array, 'X-ray Percentage':x_ray_percentage_array, 
                   'X-ray Critical Energy':x_ray_crit_energy_array,'No. X-ray Photons':no_x_ray_photons_array,
                   'No. UV Photons':no_uv_photons_array,'No. Other Photons':no_other_photons_array,
                   'Total no. Photons':total_no_photons_array,'Total Charge':total_charge_array ,'Set Emittance':set_emittance_list, 
                   'Set Radius':set_radius_list})

# Specify the file path
file_path = 'output_test.csv'

# Check if the file already exists
if os.path.exists(file_path):
    # If the file exists, append to it (without including the header)
    df.to_csv(file_path, mode='a', header=False, index=False)
else:
    # If the file does not exist, create the file and write the DataFrame with the header
    df.to_csv(file_path, index=False)

