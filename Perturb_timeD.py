import math
import numpy as np
import background as bk
from RK import *
from constants import *

def gaussian(r, r0, sigma):
    gaussianf = math.exp(-(r - r0)**2 / sigma**2)
    dgaussianf = (2 * (r0 - r) * math.exp(-(r - r0)**2 / sigma**2)) / sigma**2
    d2gaussianf = (math.exp(-(r - r0)**2 / sigma**2) * (4 * (r0 - r)**2 - 2 * sigma**2)) / sigma**4
    return gaussianf, dgaussianf, d2gaussianf


def rhs_H(r, S_r, index, l, F_func, arg1, arg2, H_func, bkg_func):
    """
    Input
    ----------
    r: array / spatial grid
    S: variable to be integrated
    index: type of initial data (integer)
    l: harmonic index (integer)
    ............................
    Output
    -------
    rhs_H: right hand side of the system, dS/dr=rhs
    """

    lamb_r, m_r, nu_r, p_r, rho_r, cs2_r = bkg_func(r)
    n = (l - 1) * (l + 2) / 2
    
    if index == 1: # Assuming H = 0 and a given F
        F_r, dF_r, d2F_r = F_func(r, arg1, arg2)
        rhs_H = math.exp(lamb_r) / r * (math.exp(nu_r - lamb_r) * (((-1 + math.exp(lamb_r) * (1 + 4 * math.pi * r ** 2 * (p_r - rho_r))) / r) *
                dF_r + d2F_r) - math.exp(nu_r) / r ** 2 * (m_r + 4 * math.pi * p_r * r ** 3) * dF_r + \
                math.exp(nu_r) / r ** 3 * (12 * math.pi * r ** 3 * rho_r - m_r - 2 * (n + 1) * r) * F_r + (8 * math.pi * r ** 2 * (rho_r + p_r) - (n + 3) + 4 * m_r / r) * S_r )

    if index == 2: # Assuming F = 0 and a given H
        H_r = H_func(r)
        rhs_H = math.exp(lamb_r) / r * ( (8 * math.pi * r ** 2 * (rho_r + p_r) - (n + 3) + 4 * m_r / r) * S_r + 8 * math.pi * r * math.exp(nu_r) * (rho_r + p_r) * H_r / cs2_r )

    return rhs_H


def set_initial_data(index, r, radius, arg1, arg2, l, bkg_func):
    """
    Input
    ----------
    index: type of initial data
    r: array / spatial grid
    argn: arguments for initial data

    Output: u_in
    -------
    """

    nr = len(r)
    dr = r[1] - r[0]
    
    for i in range(nr):
        if r[i] - radius > 0:
            nr_int = i - 1
            break

    F = np.empty(nr)
    F_dot = np.empty(nr)
    S = np.empty(nr)
    S_dot = np.empty(nr)
    H = np.empty(nr)
    H_dot = np.empty(nr)

    for i in range(nr): # Only time-symmetric initial data will be considered
        F_dot[i] = 0
        S_dot[i] = 0
        H_dot[i] = 0

    if index == 1:
        for i in range(nr):
            H[i] = 0
            F[i] = gaussian(r[i], arg1, arg2)[0]
        S[0] = 0
        for i in range(0, nr-1):
            S[i+1] = rk4_step(r[i], S[i], dr, rhs_H, index, l, gaussian, arg1, arg2, 0, bkg_func)

    elif index == 2:                 
        def H_func(r):
            if r < radius:
                return bkg_func(r)[-1] * (r / radius) ** l * math.cos( math.pi * r / (2 * radius)) 
            elif r >= radius:
                return 0
        for i in range(nr):
            H[i] = H_func(r[i])
            F[i] = 0

        #n = (l - 1) * (l + 2) / 2
        S[0] = 0 # não é exato. mudar?
        for i in range(0, nr-1):
            S[i+1] = rk4_step(r[i], S[i], dr, rhs_H, index, l, 0, 0, 0, H_func, bkg_func)

    uin = np.block([[S], [S_dot], [F], [F_dot], [H], [H_dot]]).T
    return uin


def rhs_perturb(t, u, l, lamb, m, nu, p, rho, cs2, r):
    """
    Input
    ----------
    u : numpy.array: (S, dS/dt, F, dF/dt, H, dH/dt)
    l: harmonic index (integer)
    lamb: g_{11} = Exp(lamb) (array)
    m: mass function (array)
    nu: g_{00} = - Exp(nu) (array)
    rho: energy density (array)
    p: pressure (array)
    cs2: sound speed squared (array)

    Output
    -------
    rhs : numpy.array
        Lado direito do sistema: du/dt = rhs
    """
    S = u.T[0]
    S_dot = u.T[1]
    F = u.T[2]
    F_dot = u.T[3]
    H = u.T[4]
    H_dot = u.T[5]

    nr = len(S)
    for i in range(nr):
        if p[i] == 0:
            nr_int = i
            break
        
    rhs = np.empty((nr, 6))

    dr = r[1] - r[0]

    n = (l - 1) * (l + 2) / 2

    for i in range(n_special):
        S[i] = S[n_special] * (r[i] / r[n_special]) ** (l+1)
        F[i] = F[n_special] * (r[i] / r[n_special]) ** (l+1)
        H[i] = H[n_special] * (r[i] / r[n_special]) ** l


    rhs[0] = np.array([S_dot[0],
                   math.exp(nu[0] - lamb[0]) * (((-1 + math.exp(lamb[0]) * (1 + 4 * math.pi * r[0] ** 2 * (p[0] - rho[0]))) / r[0]) *
                   ((- 0.5 * S[2] + 2 * S[1] - 1.5 * S[0]) / dr) + (- S[3] + 4 * S[2] - 5 * S[1] + 2 * S[0]) / (dr ** 2)) + 2 *                                math.exp(nu[0]) / r[0] ** 3 * (2 * math.pi * r[0] ** 3 * (rho[0] + 3 * p[0]) + m[0] ) * S[0] - math.exp(nu[0]) * (- S[3]                    + 4 * S[2] - 5 * S[1] + 2 * S[0])  / (dr ** 2)  + 4 * math.exp(2 * nu[0]) / r[0] ** 5 * ((m[0] + 4 * math.pi * p[0] *                        r[0] ** 3) ** 2 / (r[0] - 2 * m[0]) + 4 * math.pi * rho[0] * r[0] ** 3 - 3 * m[0]) * F[0],
                   F_dot[0],
                   math.exp(nu[0] - lamb[0]) * (((-1 + math.exp(lamb[0]) * (1 + 4 * math.pi * r[0] ** 2 * (p[0] - rho[0]))) / r[0]) *
                   ((- 0.5 * F[2] + 2 * F[1] - 1.5 * F[0]) / dr) + (- F[3] + 4 * F[2] - 5 * F[1] + 2 * F[0]) / (dr ** 2)) + 2 *                                math.exp(nu[0]) / r[0] ** 3 * (2 * math.pi * r[0] ** 3 * (3 * rho[0] + p[0]) + m[0] -l*(l+1)*r[0]/2) * F[0]                                    + 2 * (4 * math.pi * r[0] ** 2 * (p[0] + rho[0]) - math.exp(-lamb[0])) * S[0] + 8 * math.pi * (rho[0] + p[0]) * r[0] *                      math.exp(nu[0]) * (1 - 1 / cs2[0]) * H[0],
                   H_dot[0],
                   cs2[0] * math.exp(nu[0] - lamb[0]) * (((-1 + math.exp(lamb[0]) * (1 + 4 * math.pi * r[0] ** 2 *(p[0] - rho[0]))) / r[0])*                    ((- 0.5 * H[2] + 2 * H[1] - 1.5 * H[0]) / dr) + (- H[3] + 4 * H[2] - 5 * H[1] + 2 * H[0]) / (dr ** 2)) +                                    math.exp(nu[0]) / r[0] ** 2 * ((m[0] + 4 * math.pi * p[0] * r[0] ** 3) * (cs2[0] - 1) + 2 * cs2[0] * (r[0] - 2 * m[0])) *                    (- 0.5 * H[2] + 2 * H[1] - 1.5 * H[0]) / dr + 2 * math.exp(nu[0]) / r[0] ** 2 * (2 * math.pi * r[0] ** 2 *
                   (rho[0] + p[0]) * (3 * cs2[0] + 1)) * H[0] - math.exp(nu[0]) * cs2[0] * (l + 1)/(l - 1) * (- H[3] + 4 * H[2] - 5 * H[1] +                    2 * H[0]) / (dr ** 2) + (m[0] + 4 * math.pi * p[0] * r[0] ** 3) * (cs2[0] - 1) / (2 * r[0]) *
                   (math.exp(nu[0]) / r[0] ** 2 * (- 0.5 * F[2] + 2 * F[1] - 1.5 * F[0]) / dr - (- 0.5 * S[2] + 2 * S[1] - 1.5 * S[0]) / dr)                    + ((m[0] + 4 * math.pi * p[0] * r[0] ** 3) ** 2 / (r[0] ** 2 * (r[0] - 2 * m[0])) * (cs2[0] + 1) - (m[0] + 4 * math.pi *                    p[0] * r[0] ** 3) / (2 * r[0] ** 2) * (cs2[0] - 1) - 4 * math.pi * cs2[0] * r[0] * (3 * p[0] + rho[0])) * S[0] +                            math.exp(nu[0]) / r[0] ** 2 * (2 * (m[0] + 4 * math.pi * p[0] * r[0] ** 3) ** 2 / (r[0] ** 2 * (r[0] - 2 * m[0])) - (m[0]
                   +4*math.pi * p[0] * r[0] ** 3) / (2*r[0]**2) * (cs2[0] - 1) - 4 * math.pi * r[0] * cs2[0] * (3 * p[0] + rho[0])) * F[0]])

    rhs[-1] = np.array([- (S[-1] - S[-2]) / dr,
                   math.exp(nu[-1] - lamb[-1]) * (((-1 + math.exp(lamb[-1])) / r[-1]) * ((S[-1] - S[-2]) / dr) + (S[-1] - 2 * S[-2] + S[-3])                    / (dr ** 2))+ 2 * math.exp(nu[-1]) / r[-1] ** 3 * (m[-1] - (n + 1) * r[-1]) * S[-1] + 4 * math.exp(2 * nu[-1]) / r[-1] **                    5 * ((m[-1]) ** 2 / (r[-1] - 2 * m[-1]) - 3 * m[-1]) * F[-1],
                   - (F[-1] - F[-2]) / dr,
                   math.exp(nu[-1] - lamb[-1]) * (((-1 + math.exp(lamb[-1])) / r[-1]) * ((F[-1] - F[-2]) / dr) + (F[-1] - 2 * F[-2] + F[-3])                    / (dr ** 2)) + 2 * math.exp(nu[-1]) / r[-1] ** 3 * (m[-1] - (n + 1) * r[-1]) * F[-1] + 2 * (- math.exp(-lamb[-1]))*S[-1],
                   H_dot[-1],
                   0])

    for i in range(n_special):
        rhs[i] = np.array([S_dot[i],
                       math.exp(nu[i] - lamb[i]) * (((-1 + math.exp(lamb[i]) * (1 + 4 * math.pi * r[i] ** 2 * (p[i] - rho[i]))) / r[i]) *
                       ((S[i+1] - S[i-1]) / (2*dr)) + (S[i+1] - 2 * S[i] + S[i-1]) / (dr ** 2)) + 2 * math.exp(nu[i]) / r[i] ** 3 * (2 *                            math.pi * r[i] ** 3 * (rho[i] + 3 * p[i]) + m[i] ) * S[i] - math.exp(nu[i]) * (S[i+1] - 2 * S[i] + S[i-1]) / (dr **                          2)  + 4 * math.exp(2 * nu[i]) / r[i] ** 5 * ((m[i] + 4 * math.pi * p[i] * r[i] ** 3) ** 2 / (r[i] - 2 * m[i]) + 4 *                          math.pi * rho[i] * r[i] ** 3 - 3 * m[i]) * F[i],
                       F_dot[i],
                       math.exp(nu[i] - lamb[i]) * (((-1 + math.exp(lamb[i]) * (1 + 4 * math.pi * r[i] ** 2 * (p[i] - rho[i]))) / r[i]) *
                       ((F[i+1] - F[i-1]) / (2*dr)) + (F[i+1] - 2 * F[i] + F[i-1]) / (dr ** 2)) + 2 * math.exp(nu[i]) / r[i] ** 3 * (2 *                            math.pi * r[i] ** 3 * (3 * rho[i] + p[i]) + m[i] -l*(l+1)*r[i]/2) * F[i] + 2 * (4 * math.pi * r[i] ** 2 * (p[i] +                            rho[i]) - math.exp(-lamb[i])) * S[i] + 8 * math.pi * (rho[i] + p[i]) * r[i]*math.exp(nu[i])* (1 - 1 / cs2[i]) * H[i],
                       H_dot[i],
                       cs2[i] * math.exp(nu[i] - lamb[i]) * (((-1 + math.exp(lamb[i]) * (1 + 4 * math.pi * r[i] ** 2 * (p[i] - rho[i]))) /                          r[i]) * ((H[i+1] - H[i-1]) / (2*dr)) + (H[i+1] - 2 * H[i] + H[i-1]) / (dr ** 2)) + math.exp(nu[i]) / r[i] ** 2 *                            ((m[i] + 4 * math.pi * p[i] * r[i] ** 3) * (cs2[i] - 1) + 2 * cs2[i] * (r[i] - 2 * m[i])) * (H[i+1] - H[i-1]) / (2 *                        dr) + 2 * math.exp(nu[i]) / r[i] ** 2 * (2 * math.pi * r[i] ** 2 * (rho[i] + p[i]) * (3 * cs2[i] + 1)) * H[i] -                              math.exp(nu[i]) * cs2[i] * (l + 1)/(l - 1) * (H[i+1] - 2 * H[i] + H[i-1]) / (dr ** 2) + (m[i] + 4 * math.pi * p[i] *                        r[i] ** 3) * (cs2[i] - 1) / (2 * r[i]) *
                       (math.exp(nu[i]) / r[i] ** 2 * (F[i+1] - F[i-1]) / (2*dr) - (S[i+1] - S[i-1]) / (2*dr)) + ((m[i] + 4 * math.pi *
                       p[i] * r[i] ** 3) ** 2 / (r[i] ** 2 * (r[i] - 2 * m[i])) * (cs2[i] + 1) - (m[i] + 4 * math.pi * p[i] * r[i] ** 3) /                          (2 * r[i] ** 2) * (cs2[i] - 1) - 4 * math.pi * cs2[i] * r[i] * (3 * p[i] + rho[i])) * S[i] + math.exp(nu[i]) / r[i]                          ** 2 * (2 * (m[i] + 4 * math.pi * p[i] * r[i] ** 3) ** 2 / (r[i] ** 2 * (r[i] - 2 * m[i])) - (m[i]+ 4 * math.pi *                            p[i] * r[i] ** 3) / (2 * r[i] ** 2) * (cs2[i] - 1) - 4 * math.pi * r[i] * cs2[i] * (3 * p[i] + rho[i])) * F[i]])

    for i in range (n_special, nr_int):
        rhs[i] = np.array([S_dot[i],
                       math.exp(nu[i] - lamb[i]) * (((-1 + math.exp(lamb[i]) * (1 + 4 * math.pi * r[i] ** 2 * (p[i] - rho[i]))) / r[i]) *
                       ((S[i+1] - S[i-1]) / (2*dr)) + (S[i+1] - 2 * S[i] + S[i-1]) / (dr ** 2)) + 2 * math.exp(nu[i]) / r[i] ** 3 * (2 *                            math.pi * r[i] ** 3 * (rho[i] + 3 * p[i]) + m[i] - (n + 1) * r[i]) * S[i] + 4 * math.exp(2 * nu[i]) / r[i] ** 5 *                            ((m[i] + 4 * math.pi * p[i] * r[i] ** 3) ** 2 / (r[i] - 2 * m[i]) + 4 * math.pi * rho[i] * r[i] ** 3 - 3 * m[i]) *                          F[i],
                       F_dot[i],
                       math.exp(nu[i] - lamb[i]) * (((-1 + math.exp(lamb[i]) * (1 + 4 * math.pi * r[i] ** 2 * (p[i] - rho[i]))) / r[i]) *
                       ((F[i+1] - F[i-1]) / (2*dr)) + (F[i+1] - 2 * F[i] + F[i-1]) / (dr ** 2)) + 2 * math.exp(nu[i]) / r[i] ** 3 * (2 *                            math.pi * r[i] ** 3 * (3 * rho[i] + p[i]) + m[i] - (n + 1) * r[i]) * F[i] + 2 * (4 * math.pi * r[i] ** 2 * (p[i] +                          rho[i]) - math.exp(-lamb[i])) * S[i] + 8 * math.pi * (rho[i] + p[i]) * r[i] * math.exp(nu[i]) * (1 - 1 / cs2[i]) *                          H[i],
                       H_dot[i],
                       cs2[i] * math.exp(nu[i] - lamb[i]) * (((-1 + math.exp(lamb[i]) * (1 + 4 * math.pi * r[i] ** 2 * (p[i] - rho[i]))) /                          r[i]) * ((H[i+1] - H[i-1]) / (2*dr)) + (H[i+1] - 2 * H[i] + H[i-1]) / (dr ** 2)) + math.exp(nu[i]) / r[i] ** 2 *                            ((m[i] + 4 * math.pi * p[i] * r[i] ** 3) * (cs2[i] - 1) + 2 * cs2[i] * (r[i] - 2 * m[i])) * (H[i+1] - H[i-1]) / (2 *                        dr) + 2 * math.exp(nu[i]) / r[i] ** 2 * (2 * math.pi * r[i] ** 2 *
                       (rho[i] + p[i]) * (3 * cs2[i] + 1) - cs2[i] * (n + 1)) * H[i] + (m[i] + 4 * math.pi * p[i] * r[i] ** 3) * (cs2[i] -                          1) / (2 * r[i]) * (math.exp(nu[i]) / r[i] ** 2 * (F[i+1] - F[i-1]) / (2*dr) - (S[i+1] - S[i-1]) / (2*dr)) + ((m[i] +                        4 * math.pi * p[i] * r[i] ** 3) ** 2 / (r[i] ** 2 * (r[i] - 2 * m[i])) * (cs2[i] + 1) - (m[i] + 4 * math.pi * p[i] *                        r[i] ** 3) / (2 * r[i] ** 2) * (cs2[i] - 1) - 4 * math.pi * cs2[i] * r[i] * (3 * p[i] + rho[i])) * S[i] +                                    math.exp(nu[i]) / r[i] ** 2 * (2 * (m[i] + 4 * math.pi * p[i] * r[i] ** 3) ** 2 / (r[i] ** 2 * (r[i] - 2 * m[i])) -                          (m[i] + 4 * math.pi * p[i] * r[i] ** 3) / (2 * r[i] ** 2) * (cs2[i] - 1) - 4 * math.pi * r[i] * cs2[i] * (3 * p[i] +                        rho[i])) * F[i]])


    for i in range(nr_int, nr-1):
        rhs[i] = np.array([S_dot[i],
                       math.exp(nu[i] - lamb[i]) * (((-1 + math.exp(lamb[i])) / r[i]) * ((S[i+1] - S[i-1]) / (2*dr)) + (S[i+1] - 2 * S[i] +                        S[i-1]) / (dr ** 2)) + 2 * math.exp(nu[i]) / r[i] ** 3 * (m[i] - (n + 1) * r[i]) * S[i] + 4 * math.exp(2 * nu[i]) /                          r[i] ** 5 * ((m[i]) ** 2 / (r[i] - 2 * m[i]) - 3 * m[i]) * F[i],
                       F_dot[i],
                       math.exp(nu[i] - lamb[i]) * (((-1 + math.exp(lamb[i])) / r[i]) * ((F[i+1] - F[i-1]) / (2*dr)) + (F[i+1] - 2 * F[i] +                        F[i-1]) / (dr ** 2))+ 2 * math.exp(nu[i]) / r[i] ** 3 * (m[i] - (n + 1) * r[i]) * F[i]+2*(-math.exp(-lamb[i]))*S[i],
                       0,
                       0])

    return rhs


def H_constraint(u, l, lamb, m, nu, p, rho, cs2, r, nr_int):
    S = u.T[0]
    F = u.T[2]
    H = u.T[4]

    nr = len(S)
    #nr_int = len(p)
    dr = r[1] - r[0]
    n = (l - 1) * (l + 2) / 2

    Hamilt = np.empty((nr, 1))

    Hamilt[0] = math.exp(nu[0] - lamb[0]) * (
                ((-1 + math.exp(lamb[0]) * (1 + 4 * math.pi * r[0] ** 2 * (p[0] - rho[0]))) / r[0]) *
                ((F[1] - F[0]) / (2 * dr)) + (F[2] - 2 * F[1] + F[0]) / (dr ** 2)) - math.exp(nu[0]) / r[0] ** 2 * (
                            m[0] + 4 * math.pi * p[0] * r[0] ** 3) * ((F[1] - F[0]) / (2 * dr)) + \
                math.exp(nu[0]) / r[0] ** 3 * (12 * math.pi * r[0] ** 3 * rho[0] - m[0] - 2 * (n + 1) * r[0]) * F[0] - \
                r[0] * math.exp(-(nu[0] + lamb[0]) / 2) * math.exp((nu[0] - lamb[0]) / 2) * \
                ((S[1] - S[0]) / (2 * dr)) + (8 * math.pi * r[0] ** 2 * (rho[0] + p[0]) - (n + 3) + 4 * m[0] / r[0]) * \
                S[0] + (8 * math.pi * r[0]) / cs2[0] * math.exp(nu[0]) * (rho[0] + p[0]) * H[0]

    Hamilt[-1] = math.exp(nu[-1] - lamb[-1]) * (((-1 + math.exp(lamb[-1])) / r[-1]) *
                                                ((F[-1] - F[-2]) / (2 * dr)) + (F[-1] - 2 * F[-2] + F[-3]) / (
                                                            dr ** 2)) - math.exp(nu[-1]) / r[-1] ** 2 * m[-1] * (
                             (F[-1] - F[-2]) / (2 * dr)) + \
                 math.exp(nu[-1]) / r[-1] ** 3 * (- m[-1] - 2 * (n + 1) * r[-1]) * F[-1] - r[-1] * math.exp(
        -(nu[-1] + lamb[-1]) / 2) * math.exp((nu[-1] - lamb[-1]) / 2) * \
                 ((S[-1] - S[-2]) / (2 * dr)) + (- (n + 3) + 4 * m[-1] / r[-1]) * S[-1]

    for i in range(nr_int):
        Hamilt[i] = math.exp(nu[i] - lamb[i]) * (
                    ((-1 + math.exp(lamb[i]) * (1 + 4 * math.pi * r[i] ** 2 * (p[i] - rho[i]))) / r[i]) *
                    ((F[i + 1] - F[i - 1]) / (2 * dr)) + (F[i + 1] - 2 * F[i] + F[i - 1]) / (dr ** 2)) - math.exp(
            nu[i]) / r[i] ** 2 * (m[i] + 4 * math.pi * p[i] * r[i] ** 3) * ((F[i + 1] - F[i - 1]) / (2 * dr)) + \
                    math.exp(nu[i]) / r[i] ** 3 * (12 * math.pi * r[i] ** 3 * rho[i] - m[i] - 2 * (n + 1) * r[i]) * F[
                        i] - r[i] * math.exp(- lamb[i]) * \
                    ((S[i + 1] - S[i - 1]) / (2 * dr)) + (
                                8 * math.pi * r[i] ** 2 * (rho[i] + p[i]) - (n + 3) + 4 * m[i] / r[i]) * S[i] + (
                                8 * math.pi * r[i]) / cs2[i] * math.exp(nu[i]) * (rho[i] + p[i]) * H[i]
    for i in range(nr_int, nr - 1):
        Hamilt[i] = math.exp(nu[i] - lamb[i]) * (((-1 + math.exp(lamb[i])) / r[i]) *
                                                 ((F[i + 1] - F[i - 1]) / (2 * dr)) + (
                                                             F[i + 1] - 2 * F[i] + F[i - 1]) / (dr ** 2)) - math.exp(
            nu[i]) / r[i] ** 2 * m[i] * ((F[i + 1] - F[i - 1]) / (2 * dr)) + \
                    math.exp(nu[i]) / r[i] ** 3 * (- m[i] - 2 * (n + 1) * r[i]) * F[i] - r[i] * math.exp(- lamb[i]) * \
                    ((S[i + 1] - S[i - 1]) / (2 * dr)) + (- (n + 3) + 4 * m[i] / r[i]) * S[i]
    return Hamilt



