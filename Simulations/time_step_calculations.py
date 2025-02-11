import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import epsilon_0, e, m_e, m_p



def driver_time(gamma_D):
  time=np.sqrt((m_p*1e6*2*gamma_D)/m_e)
  return time

def plasma_freq(density):
  omega_p=np.sqrt((density*e**2)/(m_e*epsilon_0))
  return omega_p

def witness_time(gamma_w):
  time=np.sqrt(2*gamma_w)
  return time


plasma_density=7e14
gamma_D=426
#gamma_w=294
gamma_w=6656

print("Driver time:",driver_time(gamma_D))
print("Witness time:",witness_time(gamma_w))
print("Plasma frequency:",plasma_freq(plasma_density))