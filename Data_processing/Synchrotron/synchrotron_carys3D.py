import sys
import h5py
import os
import matplotlib.pyplot as plt
from matplotlib.image import NonUniformImage
from matplotlib import cm
from scipy.interpolate import griddata
from scipy.special import kv
from scipy.integrate import quad
from scipy import integrate
from scipy import signal


import numpy as np


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


run_no=8
nu = 5/3  # order of the Bessel function
uv_max=124 #eV
uv_min=3.1 #eV
xray_max=124e3 #eV

total_e_max=124e3
total_e_min=3.1



file_numbers=["00001", "00002", "00003", "00004", "00005", "00006"]
full_energy_array=[]
full_phi_array=[]
full_theta_array=[]
full_synchrotron_array=[]
for file_number in file_numbers:
    
    filename=f"/v3d_synchrotron_{file_number}.h5" ###wherever you put the data
    file=h5py.File(filename, 'r')
    full_energy_array.append(file["Energy"][:])
    full_phi_array.append(file["Phi"][:])
    full_theta_array.append(file["Theta"][:])
    full_synchrotron_array.append(file["Synchrotron3D"][:])
    file.close()

for i in range(5, 6):
    run_no=i
    e_c=full_energy_array[run_no]
    e=full_energy_array[run_no]

    #print("E_c sum:",str(np.sum(e_c)))
    #print("E sum:",str(np.sum(e)))
    #print(e)

    photons_per_crit_energy=full_synchrotron_array[run_no]
    #print("Photons per critical energy sum:",str(np.sum(photons_per_crit_energy)))

    # Initialize an empty list to store the result arrays
    matrix = []

    # Loop through each value of omega_c
    for ec in e_c:
        array = []  # Create a new list for each omega_c
        for i in e:
            value = i/ ec  # Calculate the normalized value
            array.append(value)  # Append to the array
        matrix.append(array)  # Append the array to the matrix

    # Convert the list of arrays into a 2D NumPy array
    matrix = np.array(matrix)

    matrix_file_name='Data_processing/Synchrotron/matrices/S_matrix_'+str(run_no)+'.npy'

    # # Initialize an empty matrix to store the results of the integral

    try:
        S_matrix=np.load(matrix_file_name)
    except FileNotFoundError:
        S_matrix = np.zeros_like(matrix)

    # Apply the integral to each element in the matrix
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                upper_bound=(e[i]/e_c[j])*100
                S_matrix[i, j] = S_integral(e[i], e_c[j], upper_bound)


        np.save(matrix_file_name, S_matrix)
        S_matrix=np.load(matrix_file_name)


    # Perform matrix multiplication
    #print("S_matrix shape:",S_matrix.shape)
    #print("photons_per_crit_energy shape:",photons_per_crit_energy.shape)
    #result_matrix = np.dot(S_matrix, photons_per_crit_energy)
    result_matrix = np.einsum('ij,klj->kli', S_matrix, photons_per_crit_energy)
    normalised_result=result_matrix/np.sum(S_matrix,axis=1)
    #print("normalised result:",normalised_result)



    #print("Photons per energy sum:",str(np.sum(normalised_result)))


    # Define the directory path
    # save_dir = r'C:\Users\carys\OneDrive - The University of Manchester\University\Year 4\MPhys\mphys_docs\Python_scripts\plots\carys_synchrotron'
    save_dir = 'Data_processing/Synchrotron/plots'

    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Define the full path for saving the plot
    save_path = os.path.join(save_dir, 'synchrotron_plot.ylims_3D_phi_theta_T_'+str(run_no)+'.png')


    photons_per_theta_per_phi=np.sum(full_synchrotron_array[run_no],axis=2)

    #print("photons_per_theta_per_phi shape:",photons_per_theta_per_phi.shape)
    phi=full_phi_array[run_no]
    theta=full_theta_array[run_no]*1e3

    photons_per_energy=sum_over_theta_and_phi(normalised_result)
    save_path4 = os.path.join(save_dir, 'synchrotron_plot.ylims_3D_photons_per_energy'+str(run_no)+'.png')
    plt.figure(figsize=(10, 6))  
    plt.plot(e,photons_per_energy)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Energy eV")
    plt.ylabel("No. photons/energy")
    plt.title("Photon number per energy against energy")
    plt.ylim(1e0,1e7)
    plt.savefig(save_path4, dpi=300, bbox_inches='tight')
    #plt.show()

    #print("e array:"+str(e))
    #print("photon energies array:"+str(photons_per_energy))
    print("photons_per_energy sum:"+str(np.sum(photons_per_energy)))
    #integrate to get no. of x-ray photons
    # Mask the data to include only the range of interest

    # Define energy ranges as a list of tuples (min, max)
    energy_ranges = [
        ("X-ray", 10, 100),  # X-ray range (adjust as needed)
        ("UV", 3, 10),       # UV range
        ("Other", 0, 3)      # Other photons
    ]


    # Dictionary to store results
    photon_counts = {}

    # Compute total number of photons in the specified range
    mask_total = (e >= total_e_min) & (e <= total_e_max)
    energies_total = e[mask_total]
    photons_total = photons_per_energy[mask_total]

    total_photon_count = integrate.simpson(photons_total, x=energies_total)
    print(f"Total photons from {total_e_min} to {total_e_max}: {total_photon_count}")
    # Store total count in dictionary
    photon_counts["Total"] = total_photon_count

    # Loop over energy ranges and integrate for each range
    for label, e_min, e_max in energy_ranges:
        mask = (e >= e_min) & (e <= e_max)
        energies_integration = e[mask]
        photons_integration = photons_per_energy[mask]

        photon_count = integrate.simpson(photons_integration, x=energies_integration)
        photon_counts[label] = photon_count
        percentage = (photon_count / total_photon_count) * 100

        photon_counts[label] = (photon_count, percentage)

        print(f"{label} photons: {photon_count:.2f} ({percentage:.2f}%)")



    

    # Optional: Access results as needed
    print(photon_counts)