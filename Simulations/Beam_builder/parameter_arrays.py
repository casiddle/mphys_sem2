import numpy as np
from scipy.constants import e, m_e, epsilon_0, c

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
print("gamme:",gamma)


emittance_min=2
emittance_max=30

# Create logarithmically spaced values between 2 and 30 using the natural logarithm
log_spaced_values = np.exp(np.linspace(np.log(emittance_min), np.log(emittance_max), num=10))

print("Logarithmically spaced values (base e):", log_spaced_values)

# List of fractions/multiples to compute
fractions = [0.5,1,2,3,4,5]  # You can easily add more values to this list

# Initialize a list to store the results in the desired format
results = []

# For each emittance value, calculate the corresponding beam match radii for each fraction/multiple
for emittance in log_spaced_values:
    beam_radius = sigma_ic(emittance, gamma, kp)  # Calculate beam radius for given emittance in meters
    beam_radius_um = beam_radius * 1e6  # Convert to micrometers
    
    # Create a list of tuples: [(emittance, fraction * beam_radius) for each fraction]
    for fraction in fractions:
        results.append([emittance, fraction * beam_radius_um])

# Convert results to a NumPy array for easy manipulation (optional)
results_array = np.array(results)

# Print the 2D array of results in the desired format
print("\nEmittance and corresponding radii values:")
for row in results:
    print(f"[{row[0]:.5f}, {row[1]:.5f}]")