import sys
import h5py
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import griddata
from scipy.special import kv
from scipy.integrate import quad
from scipy import integrate
from scipy import signal
import numpy as np
import argparse
from scipy.signal import convolve2d

# Get the directory where the current script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Change the current working directory to the script directory
os.chdir(script_dir)
#print("script directory sync:"+str(script_dir))

script_dir=os.path.join(script_dir,"emittance_scan")

#print("script directory sync:"+str(script_dir))

def sum_over_theta_and_phi(synchrotron_array):
    photons_per_crit_energy = np.sum(synchrotron_array, axis=(0,1))
    return photons_per_crit_energy

def integrand(x):
    return kv(nu, x)

def S_integral(E, E_c, upper_bound):
    ratio = E / E_c
    array=np.linspace(ratio,upper_bound)
    integrand_array=integrand(array)
    # Perform integration for each scalar value of ratio
    #integral_result = integrate.simpson(integrand_array, x=array)
    integral_result, _ = quad(integrand, ratio, upper_bound)
    result = ratio * integral_result  # Multiply by ratio
    return result

def S_integral2(E, E_c, upper_bound):
    """
    Calculate the integral of the function S(E, E_c) = E/E_c * K_v(E/E_c) from E/E_c to upper_bound,
    where upper bound is some large number (in practice this should be inifinity) and E is energy and E_c is
    critical energy. This function uses vectorize to do this integral for an array of E values.
    """
    # Calculate the ratio of E to E_c
    ratio = E / E_c
    
    # Apply the integrand function over all elements of the array
    # np.vectorize creates a vectorized function (one that can work over entire arrays) for the integrand over the range from 'ratio' to 'upper_bound'
    integrand_array = np.vectorize(integrand)(np.linspace(ratio, upper_bound))
    
    # Calculate the integral of the integrand function using the quad method from scipy
    # quad returns the value of the integral and an estimate of the error (which is ignored with '_')
    integral_result, _ = quad(integrand, ratio, upper_bound)
    
    # Multiply the integral result by the ratio to get the final result
    result = ratio * integral_result
    
    # Return the final result of the integral calculation
    return result

def calculate_critical_energy(energy_array, photons_per_energy_array):
    mean_energy_array=[]
    mean_energy = np.sum((energy_array*photons_per_energy_array))/np.sum(photons_per_energy_array)
    mean_energy_array.append(mean_energy)
    critical_energy_array = mean_energy_array/(8/(15*np.sqrt(3)))
    return critical_energy_array

def calculate_mean_theta(theta_array, photons_per_theta_array):
    mean_theta_array=[]
    mean_theta = np.sum((theta_array*photons_per_theta_array))/np.sum(photons_per_theta_array)
    mean_theta_array.append(mean_theta)
    # print([float(num) for num in mean_theta_array])
    return mean_theta_array

def sum_over_energy_and_phi(synchrotron_array):
    photons_per_theta = np.sum(synchrotron_array, axis = (0, 2))
    return photons_per_theta

def std_from_frequency(theta_array, photons_per_theta_array):
    # Step 1: Calculate the weighted mean
    total_frequency = sum(photons_per_theta_array)
    weighted_mean = sum(value * freq for value, freq in zip(theta_array, photons_per_theta_array)) / total_frequency
    #print("mean:",weighted_mean)
    
    # Step 2: Calculate the weighted variance
    variance = sum(freq * (value - weighted_mean) ** 2 for value, freq in zip(theta_array, photons_per_theta_array)) / total_frequency
    
    # Step 3: Take the square root of the variance
    std_dev = np.sqrt(variance)
    #print("std:",std_dev)
    return std_dev

def generate_file_numbers(num_files):
    """Generate a list of zero-padded file numbers."""
    return [f"{i:05d}" for i in range(1, num_files + 1)]

def get_file_suffix(run_no):
    """Convert a run number to a zero-padded 5-digit file suffix."""
    return f"{run_no:05d}"

run_no=10
emittance_no='fix_no_CB'
nu = 5/3  # order of the Bessel function
uv_max=124 #eV
uv_min=3.1 #eV
xray_max=124e3 #eV

middle_max=1000 #eV
middle_min=500#eV
xray_max=124e1 #eV
min=124 #eV

parser = argparse.ArgumentParser(description='Process synchrotron data and generate plots.')
parser.add_argument('--run_no', type=int, default=run_no, help='Run number to process')
parser.add_argument('--emittance',type=str, default=emittance_no, help='emittance data file number')
# Parse the command-line arguments
args = parser.parse_args()

run_no=args.run_no
emittance_no=args.emittance

if emittance_no==1.0:
    emittance_no=str(1)

print("-----------------Run-------------------------:"+str(run_no))



file_number=get_file_suffix(run_no)

full_energy_array=[]
full_phi_array=[]
full_theta_array=[]
full_synchrotron_array=[]
filename=rf"{script_dir}\emittance-{emittance_no}\v3d_synchrotron_{file_number}.h5"


print("filename:"+str(filename))
file=h5py.File(filename, 'r')
full_energy_array.append(file["Energy"][:])
full_phi_array.append(file["Phi"][:])
full_theta_array.append(file["Theta"][:])
full_synchrotron_array.append(file["Synchrotron3D"][:])
file.close()

# Convert the lists to NumPy arrays and squeeze them to remove any singleton dimensions
full_energy_array = np.squeeze(np.array(full_energy_array))
full_phi_array = np.squeeze(np.array(full_phi_array))
full_theta_array = np.squeeze(np.array(full_theta_array))
full_synchrotron_array = np.squeeze(np.array(full_synchrotron_array))

E_c=full_energy_array
E=full_energy_array



#full_synchrotron_array = np.array(full_synchrotron_array)
#full_synchrotron_array = np.squeeze(full_synchrotron_array)
photons_per_crit_energy=full_synchrotron_array



# Initialize an empty list to store the result arrays
matrix = []

# Loop through each value of omega_c
for E_c_val in E_c: #E_c_val is the critical energy from an array E_c of critical energies
        array = []  # Create a new list for each critical energy
        for E_val in E: #creates an array for each energy value
            value = E_val/ E_c_val  # Calculate the normalized value
            array.append(value)  # Append to the array
        matrix.append(array)  # Append the array to the matrix

# Convert the list of arrays into a 2D NumPy array
matrix = np.array(matrix)

matrix_file_name=f'matrices/S_matrix_10.npy'

# # Initialize an empty matrix to store the results of the integral

try:
    S_matrix=np.load(matrix_file_name)
except FileNotFoundError:
    S_matrix = np.zeros_like(matrix)
    # Create an upper bound matrix
    upper_bound_matrix = (E[:, None] / E_c) * 100  # Broadcasting to create a matrix of shape (i, j)

    # Use np.vectorize  to apply the integral to each element of the matrix
    S_matrix = np.vectorize(S_integral2)(E[:, None], E_c, upper_bound_matrix)
    np.save(matrix_file_name, S_matrix)
    S_matrix=np.load(matrix_file_name)



    np.save(matrix_file_name, S_matrix)
    S_matrix=np.load(matrix_file_name)


# Perform matrix multiplication
print('photons_per_crit_energy:',photons_per_crit_energy.shape)
print('S_matrix:',S_matrix.shape)
result_matrix = np.einsum('ij,klj->kli', S_matrix, photons_per_crit_energy)
normalised_result=result_matrix/np.sum(S_matrix,axis=1)


photons_per_theta_per_phi=np.sum(full_synchrotron_array,axis=2)


phi=full_phi_array
theta_array = full_theta_array  # Extract the array from the list
# Now you can multiply the array by 1e3
theta = theta_array * 1e3
mean_theta=np.mean(theta)

photons_per_energy=sum_over_theta_and_phi(normalised_result)


#integrate to get no. of x-ray photons
# Mask the data to include only the range of interest
mask = (E >= uv_max) 
energies_integration = E[mask]
photons_integration = photons_per_energy[mask]
x_ray_photons=integrate.simpson(photons_integration,x=energies_integration)

#integrate to get no. of UV photons
# Mask the data to include only the range of interest
mask = (E >= uv_min) & (E <= uv_max)
energies_integration = E[mask]
photons_integration = photons_per_energy[mask]
uv_photons=integrate.simpson(photons_integration,x=energies_integration)

#integrate to get no. of other photons
# Mask the data to include only the range of interest
#mask = (e >= 0) & (e <= uv_min)
mask = (E <= uv_min)
energies_integration = E[mask]
photons_integration = photons_per_energy[mask]
other_photons=integrate.simpson(photons_integration,x=energies_integration)


integral_sum=x_ray_photons+uv_photons+other_photons

x_ray_percentage=x_ray_photons/integral_sum
uv_percentage=uv_photons/integral_sum


photons_per_energy_array=np.array(photons_per_energy)
critical_energy_array = calculate_critical_energy(full_energy_array, photons_per_energy_array)


photons_per_theta = sum_over_energy_and_phi(normalised_result)
photons_per_theta_array=np.array(photons_per_theta)
#print(photons_per_theta_array)
mean_theta_array = calculate_mean_theta(full_theta_array, photons_per_theta_array)

std_dev=std_from_frequency(full_theta_array,photons_per_theta_array)
print('-'*50)
print("UV/Xray ratio:"+str(x_ray_photons/uv_photons))
#print("UV/Xray percentage ratio:"+str(uv_percentage/x_ray_percentage)) --same value as above

print("Mean Theta:"+str(mean_theta_array[0])) 
print("Critical Energy:"+str(critical_energy_array[0])) 
print('-'*50)

E=full_energy_array
theta=full_theta_array

# Mask the data to include only the X-ray energy range
xray_mask = (E >= uv_max) & (E <= xray_max)  # Mask for energies in the X-ray range

# Apply the mask to the energies and photons
xray_energies = E[xray_mask]
xray_photons_per_energy = photons_per_energy[xray_mask]

# Calculate the critical energy for X-ray photons
xray_critical_energy = calculate_critical_energy(xray_energies, xray_photons_per_energy)

# Debugging: Print the calculated critical energy
print("Critical Energy for X-ray photons:", xray_critical_energy)



# Define your energy threshold (e.g., uv_max or some other value)
energy_threshold = uv_max  # Replace with your specific threshold

# Get the energy values for the current run (full_energy_array[10])
energy_values = full_energy_array[10]  # Shape: (300,)

# Find the indices where the energy is above the threshold
valid_energy_indices = energy_values >= energy_threshold  # This gives a boolean array

# Create a new synchrotron array that excludes energies below the threshold
new_synchrotron_array = full_synchrotron_array[10][:, :, valid_energy_indices]


