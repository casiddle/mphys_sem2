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
        "python",  r"synchrotron_rewrite.py",  # Call the synchrotron_carys3D_sys_input.py script         
        "--run_no", str(save_num),"--emittance",str(emittance_num), "--radius_frac",str(radius_frac)    # Save number argument, emittance argument
    ]
    
    try:
        # Run the command and capture the output
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        # print(result.stdout)
        
        # Return the extracted properties from the output
        return extract_properties_sync(result.stdout)

    except subprocess.CalledProcessError as e:
        print(f"Error while processing  save {save_num} :")
        print("Error Output:", e.stderr)  # Print the error message
        print("Standard Output:", e.stdout)  # Also print any output captured before the error
        return None
    
import matplotlib.pyplot as plt

def plot_cumulative_x_ray_against_distance(emittance, radius, maximum_run_no):
    run_no_array = []
    x_ray_proportion = []
    
    for n in range(1, maximum_run_no):
        properties = get_properties_sync(n, emittance, radius)
        run_no_array.append(n)
        x_ray_proportion.append(properties['x_ray_percentage'])
    
    # Plot the data
    plt.plot(np.array(run_no_array)/4, x_ray_proportion, marker='o', linestyle='-')
    
    # Add title and labels
    plt.title('Cumulative X-Ray Proportion vs. Distance')
    plt.xlabel('Distance (m)')
    plt.ylabel('X-Ray Proportion')
    
    # Add gridlines
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Show the plot
    plt.show()
    return None

def plot_x_ray_against_distance(emittance, radius, maximum_run_no):
    run_no_array = []
    x_ray_proportion = []
    
    for n in range(1, maximum_run_no):
        properties = get_properties_sync(n, emittance, radius)
        run_no_array.append(n)
        if n == 1:
            x_ray_proportion.append((properties['x_ray_percentage']))
        else:
            x_ray_proportion.append((properties['x_ray_percentage']) - x_ray_proportion[n-2])
    
    # Plot the data
    plt.plot(np.array(run_no_array)/4, x_ray_proportion, marker='o', linestyle='-')
    
    # Add title and labels
    plt.title('Non-Cumulative X-Ray Proportion vs. Distance')
    plt.xlabel('Distance (m)')
    plt.ylabel('X-Ray Proportion')
    
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
plot_x_ray_against_distance(2.0, 1.0, 40)
