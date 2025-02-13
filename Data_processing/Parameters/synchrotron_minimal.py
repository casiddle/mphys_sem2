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
import scipy.signal as signal
import time

import numpy as np


def integrand(x):
    return kv(nu, x)

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




nu = 5/3  # order of the Bessel function
uv_max=124 #eV
uv_min=3.1 #eV
xray_max=124e1 #eV


array_to_convolve = np.array([4, 6 ,8])


