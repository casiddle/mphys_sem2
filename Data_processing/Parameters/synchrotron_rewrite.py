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
DEFAULT_RUN_NO=2
DEFAULT_INITIAL_EMITTANCE=1.1
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
    where upper bound is some large number (in practice this should be inifinity) and E is energy and E_c is
    critical energy.
    """
    ratio = E/E_c
    
    # Calculate the integral of the integrand function using the quad method from scipy
    # quad returns the value of the integral and an estimate of the error (which is ignored with '_')
    integral_result, _ = quad(integrand, ratio, upper_bound)

    return ratio * integral_result

def create_S_matrix(energy_array):
    """Create the matrix representing the Universal function for synchrotron-like emission"""
    temp_matrix = []
    for E_c in energy_array:
        temp_array = []
        for E in energy_array:
            temp_array.append(E/E_c)
        temp_matrix.append(temp_array)
    S_matrix = np.zeros_like(temp_matrix)
    upper_bound_matrix = (energy_array[:, None] / energy_array) * 100
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

def sum_photons(lower_boundary, upper_boundary, energy_array, photons_per_energy,):
    desired_photons = np.where(energy_array <= lower_boundary, 0, photons_per_energy)
    desired_photons = np.where(energy_array >= upper_boundary, 0, desired_photons)
    return np.sum(desired_photons)



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
    S_matrix = create_S_matrix(energy_array)
    print(S_matrix)

# Perform convolution 
photon_distribution_matrix = np.einsum('ij,klj->kli', S_matrix, synchrotron_array)
# normalised_result=result_matrix/np.sum(S_matrix,axis=1) ??????
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
print("Mean Theta:"+str(mean_theta)) 
print("Critical Energy:"+str(critical_energy)) 

print("No. X-ray photons: ",x_ray_photon_number)
print("No. UV photons: ",uv_photon_number)
print("No. Other photons: ",other_photon_number)
print("Total no. photons: ",total_photon_number)



