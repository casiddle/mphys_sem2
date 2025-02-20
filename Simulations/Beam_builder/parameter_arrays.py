import numpy as np
from scipy.constants import e, m_e, epsilon_0, c
import pandas as pd

def sigma_ic(e_n,gamma,kp):
  e_n_m=e_n*1e-6
  return (2*e_n_m**2/(gamma*kp**2))**0.25

def calc_kp(n):    
  # Plasma frequency (ω_p)
  omega_p = np.sqrt(n * e**2 / (m_e * epsilon_0))
    
  # Plasma wave number (k_p)
  k_p = omega_p / c
    
  return k_p




gamma=294
density_cm3=7e14
density=density_cm3*1e6
kp=calc_kp(density)
print('kp:',kp)
print("gamma:",gamma)


emittance_min=2
emittance_max=30

# Create logarithmically spaced values between 2 and 30 using the natural logarithm
log_spaced_values = np.exp(np.linspace(np.log(emittance_min), np.log(emittance_max), num=10))

print("Logarithmically spaced values (base e):", log_spaced_values)

# List of fractions/multiples to compute
fractions = [0.5,1,2,4,5]  # Fraction/multiple of the beam radius

# Dictionary to store results, where key is the fraction/multiple, and value is the corresponding beam radii
beam_radii_dict = {}

# For each emittance value, calculate the corresponding beam match radii for each fraction/multiple
for emittance in log_spaced_values:
    beam_radius = sigma_ic(emittance, gamma, kp)  # Calculate beam radius for given emittance in m
    beam_radius_um = beam_radius * 1e6  # Convert to micrometers
    
    # Calculate radii for each fraction/multiple and store in dictionary
    beam_radii_dict[emittance] = {fraction: fraction * beam_radius_um for fraction in fractions}

# Print the results
print("Emittance values:", log_spaced_values)
for emittance, radii in beam_radii_dict.items():
    print(f"Emittance: {emittance:.3f}")
    for fraction, radius in radii.items():
        print(f"  {fraction} * Beam Match radius: {radius:.8f} um")




#Initialize a list to store the results in the desired format
results = []

# Now, retrieve the beam radii from the dictionary without recalculating
for emittance in log_spaced_values:
    # Use the pre-calculated beam radius values from the dictionary
    beam_radius_um_values = beam_radii_dict[emittance]
    
    # Create a list of tuples: [(emittance, fraction * beam_radius) for each fraction]
    for fraction, radius in beam_radius_um_values.items():
        results.append([emittance, radius])

print("Number of results:", len(results))

# Convert results to a pandas DataFrame
df = pd.DataFrame(results, columns=["Emittance (um)", "Beam Radius (um)"])



# Optionally, save the DataFrame to a CSV file
df.to_csv(r"Simulations\Beam_builder\emittance_and_beam_radius.csv", index=False)