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

e_max=124e3
e_min=3.1



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
    save_dir = r'C:\Users\jaspe\OneDrive - The University of Manchester\Documents\Physics\MPhys_Project\mphys_docs\Python_scripts\plots\carys_synchrotron'

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
    mask = (e >= uv_max) 
    energies_integration = e[mask]
    photons_integration = photons_per_energy[mask]
    x_ray_photons=integrate.simpson(photons_integration,x=energies_integration)
    print("x-ray photons:"+str(x_ray_photons))

    #integrate to get no. of UV photons
    # Mask the data to include only the range of interest
    mask = (e >= uv_min) & (e <= uv_max)
    energies_integration = e[mask]
    photons_integration = photons_per_energy[mask]
    uv_photons=integrate.simpson(photons_integration,x=energies_integration)
    print("UV photons:"+str(uv_photons))

    #integrate to get no. of other photons
    # Mask the data to include only the range of interest
    #mask = (e >= 0) & (e <= uv_min)
    mask = (e <= uv_min)
    energies_integration = e[mask]
    photons_integration = photons_per_energy[mask]
    other_photons=integrate.simpson(photons_integration,x=energies_integration)
    print("other photons:"+str(other_photons))

    integral_sum=x_ray_photons+uv_photons+other_photons
    print("SUM:"+str(integral_sum))
    #print("X-ray percentage:"+str(x_ray_photons/integral_sum))

    # Prepare data for the pie chart
    labels = ['UV Photons', 'X-ray Photons', 'Other Photons']
    sizes = [uv_photons, x_ray_photons, other_photons]
    colors = ['#ff9999', '#66b3ff', '#99ff99']  # Colors for each section
    save_path2 = os.path.join(save_dir, 'synchrotron_proportions_'+str(run_no)+'.png')

    # Create pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=False, startangle=140)

    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title("Relative Proportions of UV, X-ray, and Other Photons at "+str(run_no)+"m")
    plt.savefig(save_path2, dpi=300, bbox_inches='tight')
    #plt.show()

    # Plotting
    plt.figure(figsize=(8, 8))
    plt.contourf( phi,theta, photons_per_theta_per_phi.T,levels=100,cmap='Greens')
    plt.colorbar(label='No. Photons')  # Optional: Add a color scale
    plt.title(r'Photon no. as a function of $\phi$ and $\theta$'+' '+str(run_no)+r'm')
    plt.ylabel(r'$\theta$(mrad)')
    plt.xlabel(r'$\phi$(rad)')
    plt.xlim(np.min(phi), np.max(phi))
    plt.ylim(np.min(theta), 0.4)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #plt.show()


    save_path3 = os.path.join(save_dir, 'synchrotron_plot.ylims_3D_phi.png')
    photons_per_phi=np.sum(full_synchrotron_array[run_no],axis=(1,2))
    plt.plot(phi,photons_per_phi)
    plt.title(r"Plot of number of photons against $\phi$"+' '+str(run_no)+r'm')
    plt.ylabel("Number of photons")
    plt.xlabel(r"$\phi$(rad)")
    plt.savefig(save_path3, dpi=300, bbox_inches='tight')
    #plt.show()
