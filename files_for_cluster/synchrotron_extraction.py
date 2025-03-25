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

# Setup script directory
# Get the directory where the current script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Change the current working directory to the script directory
os.chdir(SCRIPT_DIR)

SCRIPT_DIR=os.path.join(SCRIPT_DIR,"emittance_scan")

# Default values in case of no arguments
DEFAULT_RUN_NO=40
DEFAULT_INITIAL_EMITTANCE=2.0
DEFAULT_RADIUS=1.0 # Units of matched radius

# Useful constants
NU = 5/3 # Order of Bessel function
UV_MAX=124 # eV
UV_MIN=3.1
XRAY_MAX=124e3

def get_file_suffix(run_no):
    """Convert a run number to a zero-padded 5-digit file suffix."""
    return f"{run_no:05d}"

def integrand(x):
    return kv(NU, x)

def S_integral(E, E_c, upper_bound):
    """
    Calculate the integral of the function S(E, E_c) = E/E_c * K_v(E/E_c) from E/E_c to upper_bound,
    where upper bound is some large number (in practice this should be infinity) and E is energy and E_c is
    critical energy.
    """
    ratio = E/E_c
    
    # Calculate the integral of the integrand function using the quad method from scipy
    # quad returns the value of the integral and an estimate of the error (which is ignored with '_')
    integral_result, _ = quad(integrand, ratio, upper_bound)

    return ratio * integral_result

def create_S_matrix(energy_array):
    """Create the matrix representing the Universal function for synchrotron-like emission"""
    upper_bound_matrix = (energy_array[:, None] / energy_array) * 1000
    # Use np.vectorize  to apply the integral to each element of the matrix
    S_matrix = np.vectorize(S_integral)(energy_array[:, None], energy_array, upper_bound_matrix)
    np.save(f'S_matrix.npy', S_matrix)
    return S_matrix

def sum_over_theta_and_phi(synchrotron_array):
    photons_per_energy = np.sum(synchrotron_array, axis=(0, 1))
    return photons_per_energy

def sum_over_energy_and_phi(synchrotron_array):
    photons_per_theta = np.sum(synchrotron_array, axis = (0, 2))
    return photons_per_theta

def calculate_critical_energy(energy_array, photons_per_energy):
    energy_photons=energy_array*photons_per_energy
    mean_energy = np.sum((energy_photons))/np.sum(photons_per_energy)
    critical_energy = mean_energy/(8/(15*np.sqrt(3)))
    return critical_energy

def calculate_mean_theta(theta_array, photons_per_theta):
    mean_theta = np.sum((theta_array*photons_per_theta))/np.sum(photons_per_theta)
    return mean_theta

def sum_photons(lower_boundary, upper_boundary, energy_array, photons_per_energy):
    desired_photons = np.where(energy_array <= lower_boundary, 0, photons_per_energy)
    desired_photons = np.where(energy_array >= upper_boundary, 0, desired_photons)
    return np.sum(desired_photons)

def plot_energy_spectrum(energy_array, photons_per_energy, critical_energy):
    # Check critical energy is right
    # Find the index where energy_array[i] is the largest value <= critical_energy
    idx = np.searchsorted(energy_array, critical_energy, side='right') - 1
    
    # Calculate the width of each bin by finding the difference between consecutive edges
    energy_array = np.insert(energy_array, 0, 1)
    bin_widths = [energy_array[i+1] - energy_array[i] for i in range(len(energy_array)-1)]
    # Plot the line connecting the tops of the bars
    # The x-values will be the center of each bin for the line
    bin_centers = energy_array[:-1] + np.array(bin_widths) / 2
    plt.plot(bin_centers, photons_per_energy, linestyle='-', color='tab:red', linewidth=2)
    
    # Add gridlines with custom style
    plt.grid(True, color='grey', linestyle=':', linewidth=0.5, alpha=0.7)

    # Add a vertical line at the critical energy
    plt.axvline(x=energy_array[idx], color='tab:blue', linestyle='--', linewidth=2, label=f'Critical Energy = {critical_energy} eV')
    plt.axvline(x=critical_energy, color='tab:blue', linestyle='--', linewidth=2, label=f'Critical Energy = {critical_energy} eV')

    # Add "E_c" label near the vertical line at critical_energy
    plt.text(2*critical_energy, 3e3, r'$E_c$', color='tab:blue', fontsize=16, ha='center', va='bottom')

    # Add labels and a title
    plt.xlabel('Energy (eV)', fontsize=15)
    plt.ylabel(r'Photons / $\Delta$E', fontsize=15)
    plt.title('Photon spectrum of BR from QV3D simulation \n of AWAKE Run 2 at s=10m',fontsize=15)
    plt.xscale('log')
    plt.yscale('log')
    plt.tick_params(axis='both', labelsize=13)
    plt.xlim(1, 1e5)
    plt.ylim(1, 1e8)

    # Show the plot
    plt.show()
    return None



# Set command line arguments
parser = argparse.ArgumentParser(description='Process synchrotron data and generate plots.')
parser.add_argument('--run_no', type=int, default=DEFAULT_RUN_NO, help='Run number to process')
parser.add_argument('--emittance',type=str, default=DEFAULT_INITIAL_EMITTANCE, help='Emittance data file number')
parser.add_argument('--radius_frac', type=float, default=DEFAULT_RADIUS, help='Fraction or multiple of the critical radius')
# Parse the command-line arguments
args = parser.parse_args()
run_no = args.run_no
initial_emittance = args.emittance
radius_frac = args.radius_frac
print("Run:"+str(run_no))
print("Emittance no.:"+str(initial_emittance))
print("Radius fraction:"+str(radius_frac))

# Set file name
file_suffix = get_file_suffix(run_no)
filename = rf"{SCRIPT_DIR}/emittance-{initial_emittance}_radius-{radius_frac}/v3d_synchrotron_{file_suffix}.h5"
print(filename)

# Initialise empty arrays
energy_array=[]
phi_array=[]
theta_array=[]
synchrotron_array=[]

# Read file
file = h5py.File(filename, 'r')
energy_array.append(file["Energy"][:])
phi_array.append(file["Phi"][:])
theta_array.append(file["Theta"][:])
synchrotron_array.append(file["Synchrotron3D"][:])
file.close()

# Convert the lists to NumPy arrays and squeeze them to remove any singleton dimensions
energy_array = np.squeeze(np.array(energy_array))
phi_array = np.squeeze(np.array(phi_array))
theta_array = np.squeeze(np.array(theta_array))
synchrotron_array = np.squeeze(np.array(synchrotron_array))

# Get or create the universal function "S_matrix"
try:
    S_matrix = np.load(f'S_matrix.npy')
except FileNotFoundError:
    print("S_matrix not found, creating from energy array")


# Perform convolution 
photon_distribution_matrix = np.einsum('ij,klj->kli', S_matrix, synchrotron_array) 
photon_distribution_matrix = photon_distribution_matrix/np.sum(S_matrix,axis=1) # Normalise matrix
x_ray_distribution_matrix = np.where(energy_array <= UV_MAX, 0, photon_distribution_matrix)

# Get necessary sub-distributions
photons_per_energy = sum_over_theta_and_phi(photon_distribution_matrix)
photons_per_theta = sum_over_energy_and_phi(photon_distribution_matrix)
x_rays_per_energy = sum_over_theta_and_phi(x_ray_distribution_matrix)
x_rays_per_theta = sum_over_energy_and_phi(x_ray_distribution_matrix)

# Calculate critical energies
critical_energy = calculate_critical_energy(energy_array, photons_per_energy)
x_ray_critical_energy = calculate_critical_energy(energy_array, x_rays_per_energy)

# Calculate average thetas
mean_theta = calculate_mean_theta(theta_array, photons_per_theta)
x_ray_mean_theta = calculate_mean_theta(theta_array, x_rays_per_theta)

# Calculate photon numbers
total_photon_number = sum_photons(0, 1e10, energy_array, photons_per_energy)
x_ray_photon_number = sum_photons(UV_MAX, 1e10, energy_array, photons_per_energy)
uv_photon_number = sum_photons(UV_MIN, UV_MAX, energy_array, photons_per_energy)
other_photon_number = sum_photons(0, UV_MIN, energy_array, photons_per_energy)

# Calculate desired ratios
x_ray_uv_ratio = x_ray_photon_number/uv_photon_number
x_ray_proportion = x_ray_photon_number/total_photon_number

# Print output
print("Xray/UV ratio:"+str(x_ray_uv_ratio))
print("X-ray percentage:"+str(x_ray_proportion))
print("X-ray mean theta:"+str(x_ray_mean_theta))
print("Mean Theta:"+str(mean_theta)) 
print("Critical Energy:"+str(critical_energy)) 

print("No. X-ray photons: ",x_ray_photon_number)
print("No. UV photons: ",uv_photon_number)
print("No. Other photons: ",other_photon_number)
print("Total no. photons: ",total_photon_number)
print("Critical Energy X-ray photons:", str(x_ray_critical_energy))

# plot_energy_spectrum(energy_array, photons_per_energy, critical_energy)





