import numpy as np
import math
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline, UnivariateSpline
from constants import *
import pandas as pd

def set_eos(file):
    directory = os.getcwd()
    data_eos = np.loadtxt(directory+str("\\")+str("\\")+str("Eos_table")+str("\\")+str("\\")+str(file), skiprows=1)  # Skiprows elimina a primeria linha
    #data_eos = pd.read_csv("/content/" + str(file), skiprows=1)

    rho = []  # mass density
    pressure = []  # pressure
    epsilon = []  # energy density
    cs2 = [] # sound velocity ** 2

    N = len(data_eos)

    for i in range(N):
        rho.append(data_eos[i][0]/rho_dim)
        pressure.append(data_eos[i][1]/rho_dim)
        epsilon.append(data_eos[i][2]/rho_dim)

    logp = []
    logrho = []
    for i in range(len(pressure)):
        logp.append(math.log(pressure[i]))
        logrho.append(math.log(rho[i]))

    #for i in range(N-1):
    #    cs2.append((data_eos[i+1][1] - data_eos[i][1])/(data_eos[i+1][2] - data_eos[i][2]))
    #cs2.append(cs2[-1]) # Necessary to define last point

    
    # CubicSpline

    epsilon_pressure = interp1d(pressure, epsilon, bounds_error=False)
    pressure_epsilon = UnivariateSpline(epsilon, pressure, s=0)
    cs2_epsilon = pressure_epsilon.derivative()
    logrho_logp = UnivariateSpline(logp, logrho, s=0)
    gammainverse = logrho_logp.derivative()

    
    for i in range(len(epsilon)):
        cs2.append(cs2_epsilon(epsilon[i]))

    return epsilon_pressure, data_eos, cs2_epsilon, pressure, cs2, gammainverse, epsilon, pressure_epsilon #Returns the function performing the interpolation, the raw data, and cs2
    

#epsilon_pressure, data_eos, cs2_pressure, pressure, cs2 = set_eos("apr.csv")
