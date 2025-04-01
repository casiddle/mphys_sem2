import subprocess
import numpy as np
import os
import matplotlib.pyplot as plt

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

# r"synchrotron_carys3D_sys_input.py"
# r"synchrotron_rewrite.py"

def get_properties_sync(save_num,emittance_num,radius_frac):
    """
    Run synchrotron_carys3D_sys_input.py  with given parameters and return the extracted properties.
    """
    cmd = [
        "python",  r"synchrotron_extraction.py",  # Call the synchrotron_carys3D_sys_input.py script         
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



def plot_photon_number_against_distance(emittance, radius, maximum_run_no):
    try:
        photon_number = np.load(f'photon_numbers_to_plot.npy')
        run_no_array = np.load(f'run_numbers_to_plot.npy')
        cumulative_photon_number = np.load(f'cumulative_photon_number_to_plot.npy')
        beam_energy_array = np.load(f'beam_energy_to_plot.npy')
    except FileNotFoundError:
        run_no_array = []
        photon_number = []
        cumulative_photon_number = []
        beam_energy_array = []
        print("Photon numbers not found, creating from h5 files")
        for n in range(1, 11):
            properties = get_properties_sync(4*n, emittance, radius)
            beam_properties = get_properties_em('emittance_scan\emittance-2.0_radius-1.0', 4*n, 2)
            cumulative_photon_number.append(properties['total_no_photons'])
            beam_energy_array.append(beam_properties['beam_energy'])
            run_no_array.append(n)
            if n == 1:
                photon_number.append(cumulative_photon_number[0])
            else:
                photon_number.append(cumulative_photon_number[n-1] - cumulative_photon_number[n-2])
        photon_number = np.array(photon_number)
        cumulative_photon_number = np.array(cumulative_photon_number)
        run_no_array = np.array(run_no_array)
        beam_energy_array = np.array(beam_energy_array)
        np.save(f'photon_numbers_to_plot.npy', photon_number)
        np.save(f'cumulative_photon_number_to_plot.npy', cumulative_photon_number)
        np.save(f'run_numbers_to_plot.npy', run_no_array)
        np.save(f'beam_energy_to_plot.npy', beam_energy_array)
    
    # Plot the data
    print(run_no_array)
    print(photon_number)
    plt.plot(run_no_array-1, (photon_number/cumulative_photon_number[9])*100, marker='o', linestyle='-')

    # Plot model
    plt.plot(run_no_array-1, 0.5*np.sqrt(100*beam_energy_array), color='tab:red')
    
    # Add title and labels
    plt.title('Non-Cumulative Photon Number vs. Distance')
    plt.xlabel('Distance (m)')
    plt.ylabel('Photon percentage')
    
    # Add gridlines
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Show the plot
    plt.show()
    return None

# Get the directory where the current script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Change the current working directory to the script directory
os.chdir(script_dir)

# plot_cumulative_x_ray_against_distance(2.0, 1.0, 40)
plot_photon_number_against_distance(2.0, 1.0, 40)
