import math
import numpy as np
import matplotlib.pyplot as plt
from constants import *
from EOS import set_eos
from RK import rk4_step


def rhs(r, u, ep):
    # u'(r) = rhs(u,r)
    # λ, v, p = u
    rhs = np.array([(1-math.exp(u[0])+math.exp(u[0])*(r**2)*8*math.pi*ep(u[2]))/r, (-1+math.exp(u[0])+math.exp(u[0])*(r**2)*8*math.pi*u[2])/r, -(1/2)*(u[2]+ep(u[2]))*((-1+math.exp(u[0])+math.exp(u[0])*(r**2)*8*math.pi*u[2])/r)])
    return rhs


def solve_TOV(file, pc, dr):
    rf = rf_TOV
    N = int(rf / dr)

    u = np.empty((N+1, 3))

    # Set EOS
    ep_eos = set_eos(file)[0]
    p_surface = (set_eos(file)[1])[0][1]/rho_dim

    #Initial Conditions
    λ0 = 0
    ε = ri_TOV
    λε = λ0 + (8.0/3.0) * math.pi * ep_eos(pc) * ε**2
    v0 = 1
    u[0] = np.array([λε, v0, pc])

    r = np.arange(ε, rf, dr)

    for i in range(N-1):
        v = [v0]
        u[i+1] = rk4_step(r[i], u[i], dr, rhs, ep_eos) #RK4
        if u[i+1][2] < 0 or math.isnan(u[i+1][2]):
            index = i
            break
    # BINARY SEARCH

    raio_bs = r[index]
    ui = u[index]  # list with λ, v, p evaluated on the radius of the star
    dr = dr / 2.0
    while ui[2] > max(p_surface, 10**(-16)):  # ui[2] = pressure
        uaux = rk4_step(raio_bs, ui, dr, rhs, ep_eos)  # RK4
        if uaux[2] > 0.0:
            raio_bs = raio_bs + dr
            ui = uaux
        else: # If p < 0 or nan, decrease the step.
            dr = dr/2.0
    mass = (1 - math.exp(-ui[0])) * raio_bs / 2
    c_nu = np.log(1 - 2 * mass/raio_bs) - ui[1]

    return raio_bs, mass, u[:index,0], u[:index,1] + c_nu, u[:index,2], r[:index] # raio, massa, lambda, nu, p, raios

def mass_radius(file, dr):
    p_max = (set_eos(file)[1])[-1][1]/rho_dim
    pc_1 = np.linspace(0.01, p_max/4, 500)
    pc_2 = np.linspace(p_max/4, p_max, 200)
    pc = [*pc_1, *pc_2]
    r_pc = []
    m_pc = []
    for i in range(len(pc)):
        r_pc.append(solve_TOV(file, pc[i], dr)[0] * LightC / math.sqrt(NewtonG * rho_dim) / 100000)
        m_pc.append(solve_TOV(file, pc[i], dr)[1] * LightC**3 / math.sqrt(NewtonG**3 * rho_dim) /MSun)
    return pc, r_pc, m_pc


#print((set_eos("apr.csv")[1])[0][1]/rho_dim)

#plt.plot(solve_TOV("apr.csv",5)[-1],solve_TOV("apr.csv",5)[3])
#plt.show()
