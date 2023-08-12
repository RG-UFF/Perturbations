import math
import background as bk
import numpy as np
from constants import *
from scipy.interpolate import interp1d
from EOS import set_eos
from constants import *
from RK import *
from scipy.integrate import solve_ivp
from sympy import symbols, Eq, solve
from scipy.integrate import ode
from scipy.integrate import complex_ode

def BC(r0, k0, w0, nu0, p0, rho0, gamma0, l, omega, file_eos):     # Boundary Conditions

    p2 = -(2/3)*(math.pi*((3.*(((p0)**2)))+((4.*((p0)*(rho0)))+(((rho0)**2)))))
    ep_eos = set_eos(file_eos)[0]
    cs2_eos = set_eos(file_eos)[2]
    cs2_p0 = cs2_eos(ep_eos(p0))

    rho2 = p2/cs2_p0

    #u0 = np.array([0]*4)

    u0 = [0]*4

    h10 = ((2.*((l*(k0))+(8.*(math.pi*((w0)*((p0)+(rho0)))))))/(1.+l))/l

    aux0h12 = (omega ** 4.) * ((p0) * ((gamma0) * ((l * (k0)) + (8. * ( math.pi * ((w0) * ((p0) + (rho0))))))))
    aux1h12 = (3. * ((14. + ((10. * l) + (l ** 2))) * (p0))) - ((-14. + ((6. * l) + ((11. * (l ** 2))+ (2. * (l ** 3.))))) * (rho0))
    aux2h12 = (4. + ((l * (4. + (-8. * (gamma0)))) + ((5. * (gamma0 )) + (-2. * ((l ** 2) * (gamma0)))))) * (rho0)
    aux3h12 = (3. * ((((p0) ** 2)) * (1. + (l + (5. * (gamma0)))))) + (((p0) * aux2h12) + ((1. + l) * (((rho0) ** 2))))
    aux4h12 = (l * ((k0) * ((p0) * ((gamma0) * aux1h12)))) + (8. * ((2. + l) * ( math.pi * ((w0) * (((p0) + (rho0)) * aux3h12)))))
    aux5h12 = (2. * (math.pi * (((((p0) + (rho0)) ** 2))*((3. * (p0)) + (rho0)))))+(3. * ((p0) * ((gamma0) * (rho2))))
    aux6h12 = (2. + ((3. * l) + (l ** 2))) * (math.pi * (((3. * (k0))+(8. * (math.pi * ((w0) * ((3. * (p0)) + (rho0))))))*aux5h12))
    aux7h12 = (-12. * ((np.exp((nu0))) * (math.pi * ((omega ** 2) * aux4h12))))+(8. * (( np.exp((2. * (nu0)))) * (l * aux6h12)))
    aux8h12 = (0.111111 * ((np.exp((-nu0))) * ((omega ** -2.) * ((-9. * ((4. + l) * aux0h12)) + aux7h12)))) / (gamma0)

    h12 = ((((aux8h12 / (p0)) / (3. + (2. * l))) / (2. + l)) / (1. + l)) / l

    aux0k2 = (p0) * (-2. + ((3. * (l * (-1. + (gamma0)))) + (((l ** 2) * (-1. + (gamma0))) + (6. * (gamma0)))))
    aux1k2 = (l * ((6. + ((3. * l) + (l ** 2))) * ((k0) * ((p0) * (gamma0)))))+ (8. * (math.pi * ((w0) * (((p0) + (rho0)) * (aux0k2 - ((2. + ((3. * l) + (l ** 2))) * (rho0)))))))
    aux2k2 = 3. * ((((p0) ** 2)) * (2. + (l + ((6. * (gamma0)) + ((3. * (l * (gamma0))) + ((l ** 2) * (gamma0)))))))
    aux3k2 = (-7. * ((l ** 2) * (gamma0))) + ((-2. * ((l ** 3.) * ( gamma0))) + ((l * (6. + (gamma0))) + (6. * (2. + (3. * ( gamma0))))))
    aux4k2 = (2. * (math.pi * (((((p0) + (rho0)) ** 2))*((3. * (p0)) + (rho0)) )))+(3. * ((p0) * ((gamma0) * (rho2))))
    aux5k2 = ((k0) * (aux2k2 + (((p0) * (aux3k2 * (rho0))) + (3. * ((2. + l) * (((rho0) ** 2))))))) + (4. * ((2. + l) * ((w0) * aux4k2)))
    aux6k2 = (np.exp((-nu0))) * ((-3. * ((omega ** 2) * aux1k2)) + (-4. * ((np.exp((nu0))) * (l * ((1. + l) * (math.pi * aux5k2))))))

    k2 = ((((((0.166667 * aux6k2) / (gamma0)) / (p0)) / (3. + (2. * l)))/ (2. + l)) / (1. + l)) / l

    aux0w2 = (2. * (math.pi * (((((p0) + (rho0)) ** 2))*((3. * (p0)) + (rho0)))))+(3. * ((p0) * ((gamma0) * (rho2))))
    aux1w2 = (l ** 2) * ((1. + l) * (((3. * (k0)) + (8. * (math.pi * ((w0) * ((3. * (p0)) + ( rho0)))))) * aux0w2))
    aux2w2 = ((p0) + (rho0)) * (((p0) * (-2. + ((l * (-1. + (gamma0)) ) + (-6. * (gamma0))))) - ((2. + l) * (rho0)))
    aux3w2 = (2. * ((p0) * ((2. + ((-4. + ((3. * l) + (l ** 2))) * (gamma0))) * (rho0)))) + (((rho0) ** 2))
    aux4w2 = 2. * (math.pi * (((p0) + (rho0))*(((((p0) ** 2)) * (3. + (6. * ((-1. + l) * (gamma0))))) + aux3w2)))
    aux5w2 = (-3. * ((k0) * aux2w2)) + (4. * ((w0) * (aux4w2 + (-3. * ((1. + l) * ((p0) * ((gamma0) * (rho2))))))))
    aux6w2 = (2. * ((np.exp((2. * (nu0)))) * aux1w2)) + (3. * ((np.exp((nu0))) * (l * ((omega ** 2) * aux5w2))))
    aux7w2 = (omega ** -2.) * ((-18. * ((2. + l) * ((omega ** 4.) * ((w0) * ((((p0) +(rho0)) ** 2)))))) + aux6w2)
    aux8w2 = ((((0.0277778 * ((np.exp((-nu0))) * aux7w2)) / ((p0) + (rho0))) / (gamma0)) / (p0)) / (3. + (2. * l))

    w2 = aux8w2 / l

    aux0x0 = (np.exp((nu0))) * (l * ((3. * (k0)) + (8. * (math.pi * ((w0) * ((3. * (p0)) + (rho0)))))))
    aux1x0 = 0.166667 * ((np.exp((-0.5 * (nu0)))) * (((p0) + (rho0)) * ((-6. * ((omega ** 2) * (w0))) + aux0x0)))

    x0 = aux1x0 / l

    aux0x2 = (2. * (math.pi * (((((p0) + (rho0)) ** 2))*((3. * (p0)) + (rho0)))))+(3. * ((p0) * ((gamma0) * (rho2))))
    aux1x2 = ((3. * (p0)) + (rho0)) * (((3. * (k0)) + (8. * (math.pi * ((w0) * ((3. * (p0)) + (rho0))))))*aux0x2)
    aux2x2 = (14. + ((-4. * (l * (-1. + (gamma0)))) + ((11. * (gamma0)) + (-2. * ((l ** 2) * (gamma0)))))) * (rho0)
    aux3x2 = (3. * ((((p0) ** 2)) * (3. + (l + (5. * (gamma0)))))) + (((p0) * aux2x2) + ((5. + l) * (((rho0) ** 2))))
    aux4x2 = (w0) * (((3. * (((p0) ** 2))) + ((4. * ((p0) * (rho0))) + (((rho0) ** 2)))) * aux3x2)
    aux5x2 = ((p0) * ((10. + ((9. + ((-2. * l) + (-2. * (l ** 2)))) * (gamma0)))* (rho0))) + (4. * (((rho0) ** 2)))
    aux6x2 = (k0) * (((p0) + (rho0)) * (((((p0) ** 2)) * (6. + ((9. + (6. * l)) * (gamma0)))) + aux5x2))
    aux7x2 = (w0) * ((gamma0) * ((((21. + (-6. * l)) * (p0)) + ((11. + (-6. * l)) * (rho0))) * (rho2)))
    aux8x2 = (45. * (l * ((k0) * ((p0) * ((gamma0) * (rho2)))))) + (- 6. * (math.pi * ((5. * aux6x2) + (4. * ((p0) * aux7x2)))))
    aux9x2 = (w0) * (((((p0) + (rho0)) ** 2)) * (((p0) * (6. + (9. * ( gamma0)))) + (4. * (rho0))))
    aux10x2 = (((((p0) ** 2)) * (-3. + (6. * (gamma0)))) + (-2. * ((p0) * ((2. + (gamma0)) * (rho0))))) - (((rho0) ** 2))
    aux11x2 = (w0) * ((-2. * (math.pi * (((p0) + (rho0)) * aux10x2)))+(3. * ((p0) * ((gamma0) * (rho2)))))
    aux12x2 = (16. * ((l ** 2) * (math.pi * ((p0) * ((w0) * ((gamma0) * ((rho0) * ((p0) + (rho0)))))))))+((-8. * (math.pi * aux9x2)) + (l * ((3. * ((k0) * ((((p0) + (rho0)) ** 2)))) + (4. * aux11x2))))
    aux13x2 = (6. * ((np.exp((2. * (nu0)))) * (l * ((omega ** 2) * ((-80. * ((math.pi ** 2) * aux4x2)) + aux8x2))))) + (-45. * ((np.exp((nu0))) * ((omega ** 4.) * aux12x2)))
    aux14x2 = (270. * ((omega ** 6.) * ((w0) * ((((p0) + (rho0)) ** 2))))) + ((40. * ((np.exp((3. * (nu0)))) * ((l ** 2) * ((1. + l) * (math.pi * aux1x2))))) + aux13x2)
    aux15x2 = ((0.00185185 * ((np.exp((-1.5 * (nu0)))) * ((omega ** -2.) *aux14x2))) / (gamma0)) / (p0)

    x2 = (aux15x2 / (3. + (2. * l))) / l


    u0[0] = h10 + h12 * r0 ** 2
    u0[1] = k0 + k2 * r0 ** 2
    u0[2] = w0 + w2 * r0 ** 2
    u0[3] = x0 + x2 * r0 ** 2

    return np.array(u0)


def set_bkg(eos, pc):  #set_bk
    """
    Input
    ----------
    eos: Equations of state data
    pc: pressure at the center
    ............................
    Output
    -------
    lamb, nu, p, rho: interpolated functions
    """

    # SOLVING FOR THE BACKGROUND

    file_eos = eos
    ep_eos, data_eos, cs2_epsilon, pressure, cs2, gammainverse_eos, epsilon, pressure_epsilon = set_eos(file_eos)
    soltov = bk.solve_TOV(file_eos, pc, dr_TOV)

    radius = soltov[0]
    mass = soltov[1]

    lamb_interp = interp1d(np.append(soltov[-1], [radius]), np.append(soltov[2], [math.log(1 / (1 - 2 * mass / radius))]), bounds_error=False)
    nu_interp = interp1d(np.append(soltov[-1], [radius]), np.append(soltov[3], [math.log(1 - 2 * mass / radius)]), bounds_error=False)
    p_interp = interp1d(np.append(soltov[-1], [radius]), np.append(soltov[4], [10 ** (-16)]), bounds_error=False)

    def bkg_func(r):
        if r < radius:
            return float(lamb_interp(r)), float(nu_interp(r)), float(p_interp(r)), float(ep_eos(p_interp(r))), float(1/(gammainverse_eos(math.log(p_interp(r)))))

    return bkg_func



def rhs_FD(r, u, l, omega, bkg_func):

    # u = H1, K, W, X

    H1_r = u[0]
    K_r = u[1]
    W_r = u[2]
    X_r = u[3]

    lamb_r = bkg_func(r)[0]
    nu_r = bkg_func(r)[1]
    p_r = bkg_func(r)[2]
    rho_r = bkg_func(r)[3]
    gamma_r = bkg_func(r)[4]

    #Lambda, nu, p, rho e gamma são funções interpoladas

    ############ H1' ############

    pi = math.pi

    aux0H1 = (4. * ((r ** 2) * (omega ** 2))) + (-8. * ((np.exp(((lamb_r) + (nu_r)))) * (l * ((1. + l) * (pi * ((r ** 2) * (p_r)))))))
    aux1H1 = (-3. + (2. * ((np.exp((lamb_r))) * (l + ((l ** 2) + (16. * (pi * ((r ** 2) * (p_r))))))))) - ((np.exp((2. * (lamb_r)))) * (((1. + (8. * (pi * ((r ** 2) * (p_r))))) ** 2)))
    aux2H1 = (-4. * ((np.exp((lamb_r))) * ((r ** 2) * ((omega ** 2) * (K_r))))) + ((np.exp((nu_r))) * (aux1H1 * (K_r)))
    aux3H1 = ((H1_r) * (aux0H1 - ((np.exp((nu_r))) * ((-1. + (np.exp((lamb_r)))) * (l *(1. + l)))))) + ((32. * ((np.exp(((lamb_r) + (0.5 * (nu_r))))) * (pi * ((r ** 2) * (X_r))))) + aux2H1)
    aux4H1 = (0.5 * ((np.exp((-nu_r))) * aux3H1)) / (-3. + ((np.exp((lamb_r))) * (1. + (l + ((l ** 2) + (8. * (pi * ((r ** 2) * (p_r)))))))))
    aux5H1 = (-1. + ((np.exp((lamb_r))) * (1. + (8. * (pi * ((r ** 2) * (p_r))))))) * ((W_r)* ((p_r) + (rho_r)))
    aux6H1 = (-4. * ((r ** 2) * (omega ** 2))) + ((np.exp(((lamb_r) + (nu_r)))) * (l * ((1. +l) * (1. + (8. * (pi * ((r ** 2) * (p_r))))))))
    aux7H1 = ((np.exp((2. * (lamb_r)))) * (((1. + (8. * (pi * ((r ** 2) * (p_r))))) ** 2))) + (-2. * ((np.exp((lamb_r))) * (l + ((l ** 2) + (16. * (pi * ((r ** 2) * (p_r))))))))
    aux8H1 = (4. * ((np.exp((lamb_r))) * ((r ** 2) * ((omega ** 2) * (K_r))))) + ((np.exp((nu_r))) * ((3. + aux7H1) * (K_r)))
    aux9H1 = ((H1_r) * (aux6H1 - ((np.exp((nu_r))) * (l * (1. + l))))) + ((-32. * ((np.exp(((lamb_r) + (0.5 * (nu_r))))) * (pi * ((r ** 2) * (X_r))))) + aux8H1)
    aux10H1 = (0.25 * ((np.exp((-0.5 * (nu_r)))) * (aux9H1 * ((p_r) + (rho_r))))) / (-3.+ ((np.exp((lamb_r))) * (1. + (l + ((l ** 2) + (8. * (pi * ((r ** 2) * (p_r)))))))))
    aux11H1 = (X_r) + ((-0.5 * ((np.exp((0.5 * ((nu_r) - (lamb_r))))) * ((r ** -2.) *aux5H1))) + aux10H1)
    aux12H1 = ((H1_r) * (-1. + (4. * (pi * ((r ** 2) * ((rho_r) - (p_r))))))) + (-16. * ((np.exp((0.5 * (nu_r)))) * (pi * ((omega ** -2.) * aux11H1))))

    dH1=(((np.exp((lamb_r))) * ((K_r) + (aux4H1 + aux12H1))) - (l * (H1_r))) / r

    H1l = dH1


    ############ K' ############

    aux0K = (4. * ((r ** 2) * (omega ** 2))) + (-8. * ((np.exp(((lamb_r) + (nu_r)))) * (l * ((1. + l) * (pi * ((r ** 2) * (p_r)))))))
    aux1K = (-3. + (2. * ((np.exp((lamb_r))) * (l + ((l ** 2) + (16. * (pi * ((r ** 2) * (p_r))))))))) - ((np.exp((2. * (lamb_r)))) * (((1. + (8. * (pi * ((r ** 2) * (p_r))))) ** 2)))
    aux2K = (-4. * ((np.exp((lamb_r))) * ((r ** 2) * ((omega ** 2) * (K_r))))) + ((np.exp((nu_r))) * (aux1K * (K_r)))
    aux3K = ((H1_r) * (aux0K - ((np.exp((nu_r))) * ((-1. + (np.exp((lamb_r)))) * (l * (1. + l)))))) + ((32. * ((np.exp(((lamb_r) + (0.5 * (nu_r))))) * (pi * ((r ** 2) * (X_r))))) + aux2K)
    aux4K = ((np.exp((-nu_r))) * aux3K) / (-3. + ((np.exp((lamb_r))) * (1. + (l + ((l ** 2) + (8. * (pi * ((r ** 2) * (p_r)))))))))
    aux5K = ((np.exp((lamb_r))) * ((1. + (8. * (pi * ((r ** 2) * (p_r))))) * (K_r))) + (aux4K + (-16. * ((np.exp((0.5 * (lamb_r)))) * (pi * ((W_r) * ((p_r) + (rho_r)))))))

    dK=(0.5 * (((l * ((1. + l) * (H1_r))) + aux5K) - ((3. + (2. * l)) * (K_r)))) / r

    Kl = dK

    ############ W' ############

    aux0W = (r ** 2) * ((omega ** 2) * (-5. + ((np.exp((lamb_r))) * (1. + (8. * (pi * ((r ** 2) *(p_r))))))))
    aux1W = (l ** 2) * ((((1. + l) ** 2)) * (-1. + ((np.exp((lamb_r))) * (1. + (8. * (pi * ((r ** 2) * (p_r))))))))
    aux2W = (-4. * ((r ** 4.) * (omega ** 4.))) + (((np.exp((nu_r))) * (l * ((1. + l) * aux0W))) + ((np.exp((2. * (nu_r)))) * aux1W))
    aux3W = (-2. * ((r ** 2) * (omega ** 2))) + ((np.exp(((lamb_r) + (nu_r)))) * (l * (1. + (8. * (pi * ((r ** 2) * (p_r)))))))
    aux4W = (-3. + ((np.exp((lamb_r))) * (1. + (l + ((l ** 2) + (8. * (pi * ((r ** 2) * (p_r))))))))) * (W_r)
    aux5W = ((np.exp((2. * (lamb_r)))) * (((1. + (8. * (pi * ((r ** 2) * (p_r))))) ** 2))) + (-2. * ((np.exp((lamb_r))) * (l + ((l ** 2) + (16. * (pi * ((r ** 2) * (p_r))))))))
    aux6W = ((np.exp((2. * (lamb_r)))) * (((1. + (8. * (pi * ((r ** 2) * (p_r))))) ** 2))) + (-2. * ((np.exp((lamb_r))) * (2. + (l + ((l ** 2) + (32. * (pi * ((r ** 2) * (p_r)))))))))
    aux7W = ((np.exp((2. * (nu_r)))) * (l * ((1. + l) * (3. + aux5W)))) + ((np.exp((nu_r))) * ((r ** 2) * ((omega ** 2) * (15. + aux6W))))
    aux8W = (np.exp((0.5 * (lamb_r)))) * ((r ** 2) * (((4. * ((np.exp((lamb_r))) * ((r ** 4.) * (omega ** 4.)))) + aux7W) * (K_r)))
    aux9W = (-2. * ((np.exp((nu_r))) * ((1. + l) * ((aux3W - ((np.exp((nu_r))) * l)) *aux4W)))) + aux8W
    aux10W = (p_r) * ((gamma_r) * (((np.exp((0.5 * (lamb_r)))) * ((r ** 2) * ((H1_r) *aux2W))) + aux9W))
    aux11W = (np.exp((lamb_r))) * (1. + (l + ((l ** 2) + (8. * (pi * ((r ** 2) * ((p_r) * (1. + (gamma_r)))))))))
    aux12W = (gamma_r) * (-3. + ((np.exp((lamb_r))) * (1. + (l + ((l ** 2) + (-8. * (pi * ((r ** 2) * (rho_r)))))))))
    aux13W = ((r ** 2) * ((omega ** 2) * ((-3. + aux11W) * ((p_r) + (rho_r))))) - ((np.exp((nu_r))) * (l * ((1. + l) * ((p_r) * aux12W))))
    aux14W = ((r ** -2.) * aux10W) + ((-4. * ((np.exp((0.5 * ((lamb_r) + (nu_r))))) * ((X_r) * aux13W))) / ((p_r) + (rho_r)))
    aux15W = ((-0.25 * ((np.exp((-nu_r))) * ((omega ** -2.) * aux14W))) / (gamma_r)) / (-3. + ((np.exp((lamb_r))) * (1. + (l + ((l ** 2) + (8. * (pi * ((r ** 2) * (p_r)))))))))

    dW=(aux15W / (p_r)) / r

    Wl = dW

    ############ X' ############

    aux0X = 24. * ((np.exp((0.5 * ((lamb_r) + (nu_r))))) * (l * ((r ** 4.) * ((omega ** 2) * (X_r)))))
    aux1X = -8. * ((np.exp((0.5 * ((3. * (lamb_r)) + (nu_r))))) * (l * ((r ** 4.) * ((omega ** 2) * (X_r)))))
    aux2X = (np.exp((0.5 * ((3. * (lamb_r)) + (nu_r))))) * ((l ** 2) * ((r ** 4.) * ((omega ** 2) * (X_r))))
    aux3X = (np.exp((0.5 * ((3. * (lamb_r)) + (nu_r))))) * ((l ** 3.) * ((r ** 4.) * ((omega ** 2) * (X_r))))
    aux4X = (((np.exp((nu_r))) * (l * (1. + l))) - ((r ** 2) * (omega ** 2))) * (((p_r) ** 4.)* ((-2. * (W_r)) + ((np.exp((0.5 * (lamb_r)))) * ((r ** 2) * (K_r)))))
    aux5X = -6. * ((np.exp(((2. * (lamb_r)) + (nu_r)))) * ((r ** 2) * ((omega ** 2) * ((W_r)* (rho_r)))))
    aux6X = -2. * ((np.exp(((3. * (lamb_r)) + (nu_r)))) * ((r ** 2) * ((omega ** 2) * ((W_r)* (rho_r)))))
    aux7X = 14. * ((np.exp(((lamb_r) + (nu_r)))) * (l * ((r ** 2) * ((omega ** 2) * ((W_r) * (rho_r))))))
    aux8X = (np.exp(((2. * (lamb_r)) + (nu_r)))) * (l * ((r ** 2) * ((omega ** 2) * ((W_r) * (rho_r)))))
    aux9X = (np.exp(((3. * (lamb_r)) + (nu_r)))) * (l * ((r ** 2) * ((omega ** 2) * ((W_r) * (rho_r)))))
    aux10X = (np.exp(((lamb_r) + (nu_r)))) * ((l ** 2) * ((r ** 2) * ((omega ** 2) * ((W_r) *(rho_r)))))
    aux11X = (np.exp(((2. * (lamb_r)) + (nu_r)))) * ((l ** 2) * ((r ** 2) * ((omega ** 2) * ((W_r) * (rho_r)))))
    aux12X = (np.exp(((3. * (lamb_r)) + (nu_r)))) * ((l ** 2) * ((r ** 2) * ((omega ** 2) * ((W_r) * (rho_r)))))
    aux13X = -8. * ((np.exp((2. * (lamb_r)))) * (l * ((r ** 4.) * ((omega ** 4.) * ((W_r) * (rho_r))))))
    aux14X = -8. * ((np.exp((2. * (lamb_r)))) * ((l ** 2) * ((r ** 4.) * ((omega ** 4.) * ((W_r) * (rho_r))))))
    aux15X = -32. * ((np.exp((1.5 * ((lamb_r) + (nu_r))))) * (l * (pi * ((r ** 4.) * ((X_r) *(rho_r))))))
    aux16X = (np.exp(((2.5 * (lamb_r)) + (1.5 * (nu_r))))) * (l * (pi * ((r ** 4.) * ((X_r) *(rho_r)))))
    aux17X = (np.exp((1.5 * ((lamb_r) + (nu_r))))) * ((l ** 2) * (pi * ((r ** 4.) * ((X_r) * (rho_r)))))
    aux18X = (np.exp(((2.5 * (lamb_r)) + (1.5 * (nu_r))))) * ((l ** 2) * (pi * ((r ** 4.) * ((X_r) * (rho_r)))))
    aux19X = (np.exp((0.5 * ((3. * (lamb_r)) + (nu_r))))) * (pi * ((r ** 6.) * ((omega ** 2)* ((X_r) * (rho_r)))))
    aux20X = (np.exp((0.5 * ((5. * (lamb_r)) + (nu_r))))) * (pi * ((r ** 6.) * ((omega ** 2)* ((X_r) * (rho_r)))))
    aux21X = -3. * ((np.exp(((1.5 * (lamb_r)) + (2. * (nu_r))))) * (l * ((r ** 2) * ((K_r) * (rho_r)))))
    aux22X = 3. * ((np.exp(((0.5 * (lamb_r)) + (2. * (nu_r))))) * ((l ** 2) * ((r ** 2) * ((K_r) * (rho_r)))))
    aux23X = -5. * ((np.exp(((1.5 * (lamb_r)) + (2. * (nu_r))))) * ((l ** 2) * ((r ** 2) * ((K_r) * (rho_r)))))
    aux24X = 3. * ((np.exp(((2.5 * (lamb_r)) + (2. * (nu_r))))) * ((l ** 2) * ((r ** 2) * ((K_r) * (rho_r)))))
    aux25X = (np.exp(((1.5 * (lamb_r)) + (2. * (nu_r))))) * ((l ** 3.) * ((r ** 2) * ((K_r) *(rho_r))))
    aux26X = (np.exp(((2.5 * (lamb_r)) + (2. * (nu_r))))) * ((l ** 3.) * ((r ** 2) * ((K_r) *(rho_r))))
    aux27X = (np.exp(((1.5 * (lamb_r)) + (2. * (nu_r))))) * ((l ** 4.) * ((r ** 2) * ((K_r) *(rho_r))))
    aux28X = (np.exp(((2.5 * (lamb_r)) + (2. * (nu_r))))) * ((l ** 4.) * ((r ** 2) * ((K_r) *(rho_r))))
    aux29X = 21. * ((np.exp(((0.5 * (lamb_r)) + (nu_r)))) * ((r ** 4.) * ((omega ** 2) * ((K_r) * (rho_r)))))
    aux30X = -25. * ((np.exp(((1.5 * (lamb_r)) + (nu_r)))) * ((r ** 4.) * ((omega ** 2) * ((K_r) * (rho_r)))))
    aux31X = 3. * ((np.exp(((2.5 * (lamb_r)) + (nu_r)))) * ((r ** 4.) * ((omega ** 2) * ((K_r) * (rho_r)))))
    aux32X = pi * ((r ** 2) * ((((np.exp((nu_r))) * (l * (1. + l))) - ((r ** 2) * (omega ** 2)))* (rho_r)))
    aux33X = (-3. * ((np.exp((lamb_r))) * ((r ** 2) * (omega ** 2)))) + (8. * ((np.exp((lamb_r))) * aux32X))
    aux34X = (3. * ((np.exp(((lamb_r) + (nu_r)))) * (l * (1. + l)))) + (((r ** 2) * (omega ** 2)) + aux33X)
    aux35X = (np.exp((0.5 * (lamb_r)))) * ((r ** 2) * ((K_r) * ((-5. * ((np.exp((nu_r))) * (l * (1. + l)))) + aux34X)))
    aux36X = pi * ((r ** 2) * ((((np.exp((nu_r))) * (l * (1. + l))) - ((r ** 2) * (omega ** 2)))* (rho_r)))
    aux37X = ((np.exp(((lamb_r) + (nu_r)))) * (l * (3. + ((4. * l) + ((2. * (l ** 2)) + (l ** 3.)))))) + (((r ** 2) * (omega ** 2)) + (8. * ((np.exp((lamb_r))) * aux36X)))
    aux38X = ((-5. * ((np.exp((nu_r))) * (l * (1. + l)))) + aux37X) - ((np.exp((lamb_r))) * ((3. + (l + (l ** 2))) * ((r ** 2) * (omega ** 2))))
    aux39X = (np.exp(((2. * (lamb_r)) + (nu_r)))) * ((pi ** 2) * ((r ** 4.) * (((p_r) ** 3.)* (aux35X + (-2. * ((W_r) * aux38X))))))
    aux40X = -2. * ((np.exp(((lamb_r) + (2. * (nu_r))))) * (l * (5. + ((6. * l) + ((2. * (l ** 2)) + (l ** 3.))))))
    aux41X = (np.exp((2. * ((lamb_r) + (nu_r))))) * (l * (3. + ((5. * l) + ((4. * (l ** 2)) + (2. * (l ** 3.))))))
    aux42X = ((np.exp(((lamb_r) + (nu_r)))) * (l * (3. + ((4. * l) + ((2. * (l ** 2)) + (l ** 3.)))))) + ((r ** 2) * (omega ** 2))
    aux43X = ((-5. * ((np.exp((nu_r))) * (l * (1. + l)))) + aux42X) - ((np.exp((lamb_r))) * ((3. + (l + (l ** 2))) * ((r ** 2) * (omega ** 2))))
    aux44X = (-4. * ((np.exp((lamb_r))) * ((r ** 4.) * (omega ** 4.)))) + (8. * ((np.exp(((lamb_r) + (nu_r)))) * (pi * ((r ** 2) * (aux43X * (rho_r))))))
    aux45X = (-2. * ((np.exp(((lamb_r) + (nu_r)))) * ((1. + (l + (l ** 2))) * ((r ** 2) * (omega ** 2))))) + aux44X
    aux46X = (7. * ((np.exp((2. * (nu_r)))) * (l * (1. + l)))) + (aux40X + (aux41X + ((13. * ((np.exp((nu_r))) * ((r ** 2) * (omega ** 2)))) + aux45X)))
    aux47X = (np.exp(((2. * (lamb_r)) + (nu_r)))) * ((3. + ((2. * l) + (2. * (l ** 2)))) * ((r ** 2) * (omega ** 2)))
    aux48X = 32. * ((np.exp(((lamb_r) + (0.5 * (nu_r))))) * (pi * ((r ** 4.) * ((omega ** 2)* (X_r)))))
    aux49X = 2. * ((np.exp(((lamb_r) + (2. * (nu_r))))) * (l * (3. + ((4. * l) + ((2. * (l ** 2)) + (l ** 3.))))))
    aux50X = (3. * ((np.exp(((lamb_r) + (nu_r)))) * (l * (1. + l)))) + (((r ** 2) * (omega ** 2)) + (-3. * ((np.exp((lamb_r))) * ((r ** 2) * (omega ** 2)))))
    aux51X = (np.exp(((lamb_r) + (nu_r)))) * (pi * ((r ** 2) * (((-5. * ((np.exp((nu_r))) * (l * (1. + l)))) + aux50X) * (rho_r))))
    aux52X = (3. * ((np.exp(((2. * (lamb_r)) + (nu_r)))) * ((r ** 2) * (omega ** 2)))) + ((4. * ((np.exp((lamb_r))) * ((r ** 4.) * (omega ** 4.)))) + (-8. * aux51X))
    aux53X = (-13. * ((np.exp((nu_r))) * ((r ** 2) * (omega ** 2)))) + ((2. * ((np.exp(((lamb_r) + (nu_r)))) * ((r ** 2) * (omega ** 2)))) + aux52X)
    aux54X = (-7. * ((np.exp((2. * (nu_r)))) * (l * (1. + l)))) + ((-3. * ((np.exp((2. * ((lamb_r) + (nu_r))))) * (l * (1. + l)))) + (aux49X + aux53X))
    aux55X = (-2. * ((W_r) * (aux46X - aux47X))) + ((np.exp((0.5 * (lamb_r)))) * ((r ** 2)* (aux48X - ((K_r) * aux54X))))
    aux56X = (-6. * ((np.exp(((lamb_r) + (nu_r)))) * ((r ** 2) * (omega ** 2)))) + (-4. * ((np.exp((lamb_r))) * ((r ** 4.) * (omega ** 4.))))
    aux57X = (-2. * ((np.exp(((lamb_r) + (2. * (nu_r))))) * (l * (1. + l)))) + ((7. * ((np.exp((nu_r))) * ((r ** 2) * (omega ** 2)))) + aux56X)
    aux58X = ((np.exp((2. * (nu_r)))) * (l * (1. + l))) + (((np.exp((2. * ((lamb_r) + (nu_r))))) * (l * (1. + l))) + aux57X)
    aux59X = (-3. + ((np.exp((lamb_r))) * (1. + (l + (l ** 2))))) * (aux58X - ((np.exp(((2. * (lamb_r)) + (nu_r)))) * ((r ** 2) * (omega ** 2))))
    aux60X = 2. * ((np.exp(((lamb_r) + (2. * (nu_r))))) * (l * (5. + ((6. * l) + ((2. * (l ** 2)) + (l ** 3.))))))
    aux61X = (np.exp(((2. * (lamb_r)) + (nu_r)))) * ((3. + ((2. * l) + (2. * (l ** 2)))) * ((r ** 2) * (omega ** 2)))
    aux62X = (2. * ((np.exp(((lamb_r) + (nu_r)))) * ((1. + (l + (l ** 2))) * ((r ** 2) * (omega ** 2))))) + (aux61X + (4. * ((np.exp((lamb_r))) * ((r ** 4.) * (omega ** 4.)))))
    aux63X = (-7. * ((np.exp((2. * (nu_r)))) * (l * (1. + l)))) + (aux60X + ((-13. * ((np.exp((nu_r))) * ((r ** 2) * (omega ** 2)))) + aux62X))
    aux64X = (np.exp((2. * ((lamb_r) + (nu_r))))) * (l * (3. + ((5. * l) + ((4. * (l ** 2)) + (2. * (l ** 3.))))))
    aux65X = (W_r) * (aux59X + (-8. * ((np.exp((lamb_r))) * (pi * ((r ** 2) * ((aux63X -aux64X) * (rho_r)))))))
    aux66X = pi * ((r ** 2) * ((((np.exp((nu_r))) * (l * (1. + l))) - ((r ** 2) * (omega ** 2)))* (rho_r)))
    aux67X = (3. * ((np.exp((nu_r))) * (l * (1. + l)))) + (((3. + (-2. * l)) * ((r ** 2) * (omega ** 2))) + (8. * ((np.exp((lamb_r))) * aux66X)))
    aux68X = (aux67X - ((np.exp((lamb_r))) * ((r ** 2) * (omega ** 2)))) - ((np.exp(((lamb_r) + (nu_r)))) * (l * (1. + ((2. * l) + ((2. * (l ** 2)) + (l ** 3.))))))
    aux69X = (np.exp((2. * ((lamb_r) + (nu_r))))) * (l * (1. + ((3. * l) + ((4. * (l ** 2)) + (2. * (l ** 3.))))))
    aux70X = 2. * ((np.exp(((lamb_r) + (2. * (nu_r))))) * (l * (3. + ((4. * l) + ((2. * (l ** 2)) + (l ** 3.))))))
    aux71X = (3. * ((np.exp(((2. * (lamb_r)) + (nu_r)))) * ((r ** 2) * (omega ** 2)))) + (4.* ((np.exp((lamb_r))) * ((r ** 4.) * (omega ** 4.))))
    aux72X = (-13. * ((np.exp((nu_r))) * ((r ** 2) * (omega ** 2)))) + ((2. * ((np.exp(((lamb_r) + (nu_r)))) * ((r ** 2) * (omega ** 2)))) + aux71X)
    aux73X = (-7. * ((np.exp((2. * (nu_r)))) * (l * (1. + l)))) + ((-3. * ((np.exp((2. * ((lamb_r) + (nu_r))))) * (l * (1. + l)))) + (aux70X + aux72X))
    aux74X = (4. * ((np.exp((2. * (lamb_r)))) * ((r ** 4.) * (omega ** 4.)))) + (8. * ((np.exp((lamb_r))) * (pi * ((r ** 2) * (aux73X * (rho_r))))))
    aux75X = ((np.exp(((3. * (lamb_r)) + (nu_r)))) * ((r ** 2) * (omega ** 2))) + ((-12. * ((np.exp((lamb_r))) * ((r ** 4.) * (omega ** 4.)))) + aux74X)
    aux76X = (-25. * ((np.exp(((lamb_r) + (nu_r)))) * ((r ** 2) * (omega ** 2)))) + ((3. * ((np.exp(((2. * (lamb_r)) + (nu_r)))) * ((r ** 2) * (omega ** 2)))) + aux75X)
    aux77X = (3. * ((np.exp((2. * (nu_r)))) * (l * (1. + l)))) + (aux69X + ((21. * ((np.exp((nu_r))) * ((r ** 2) * (omega ** 2)))) + aux76X))
    aux78X = (np.exp(((lamb_r) + (2. * (nu_r))))) * (l * (3. + ((5. * l) + ((4. * (l ** 2)) + (2. * (l ** 3.))))))
    aux79X = (K_r) * ((aux77X - aux78X) - ((np.exp(((3. * (lamb_r)) + (2. * (nu_r))))) * (l * (1. + l))))
    aux80X = (32. * ((np.exp(((lamb_r) + (0.5 * (nu_r))))) * (pi * ((r ** 2) * ((X_r) *aux68X))))) + aux79X
    aux81X = (-8. * ((np.exp((lamb_r))) * (pi * ((r ** 2) * ((((p_r) ** 2)) * aux55X))))) +((p_r) * ((2. * aux65X) + ((np.exp((0.5 * (lamb_r)))) * ((r ** 2) * aux80X))))
    aux82X = (4. * ((np.exp((2.5 * (lamb_r)))) * ((r ** 6.) * ((omega ** 4.) * ((K_r) * (rho_r)))))) + ((-64. * aux39X) + aux81X)
    aux83X = (-12. * ((np.exp((1.5 * (lamb_r)))) * ((r ** 6.) * ((omega ** 4.) * ((K_r) * (rho_r)))))) + aux82X
    aux84X = ((np.exp(((3.5 * (lamb_r)) + (nu_r)))) * ((r ** 4.) * ((omega ** 2) * ((K_r) *(rho_r))))) + aux83X
    aux85X = (-4. * aux25X) + ((4. * aux26X) + ((-2. * aux27X) + ((2. * aux28X) + (aux29X + (aux30X + (aux31X + aux84X))))))
    aux86X = ((np.exp(((2.5 * (lamb_r)) + (2. * (nu_r))))) * (l * ((r ** 2) * ((K_r) * (rho_r))))) + (aux22X + (aux23X + (aux24X + aux85X)))
    aux87X = (3. * ((np.exp(((0.5 * (lamb_r)) + (2. * (nu_r))))) * (l * ((r ** 2) * ((K_r) * (rho_r)))))) + (aux21X + aux86X)
    aux88X = (32. * aux16X) + ((-32. * aux17X) + ((32. * aux18X) + ((96. * aux19X) + ((-32. *aux20X) + aux87X))))
    aux89X = (-8. * ((np.exp((2. * (lamb_r)))) * ((r ** 4.) * ((omega ** 4.) * ((W_r) * (rho_r)))))) + (aux13X + (aux14X + (aux15X + aux88X)))
    aux90X = (-2. * aux12X) + ((24. * ((np.exp((lamb_r))) * ((r ** 4.) * ((omega ** 4.) * ((W_r) * (rho_r)))))) + aux89X)
    aux91X = aux6X + (aux7X + ((-12. * aux8X) + ((-2. * aux9X) + ((14. * aux10X) + ((-12. *aux11X) + aux90X)))))
    aux92X = (50. * ((np.exp(((lamb_r) + (nu_r)))) * ((r ** 2) * ((omega ** 2) * ((W_r) * (rho_r)))))) + (aux5X + aux91X)
    aux93X = (2. * ((np.exp(((3. * (lamb_r)) + (2. * (nu_r))))) * ((l ** 4.) * ((W_r) * (rho_r))))) + ((-42. * ((np.exp((nu_r))) * ((r ** 2) * ((omega ** 2) * ((W_r) * (rho_r)))))) + aux92X)
    aux94X = (-4. * ((np.exp((2. * ((lamb_r) + (nu_r))))) * ((l ** 4.) * ((W_r) * (rho_r))))) + ((2. * ((np.exp(((lamb_r) + (2. * (nu_r))))) * ((l ** 4.) * ((W_r) * (rho_r)))))+ aux93X)
    aux95X = (4. * ((np.exp(((3. * (lamb_r)) + (2. * (nu_r))))) * ((l ** 3.) * ((W_r) * (rho_r))))) + aux94X
    aux96X = (-8. * ((np.exp((2. * ((lamb_r) + (nu_r))))) * ((l ** 3.) * ((W_r) * (rho_r))))) + ((4. * ((np.exp(((lamb_r) + (2. * (nu_r))))) * ((l ** 3.) * ((W_r) * (rho_r)))))+ aux95X)
    aux97X = (4. * ((np.exp(((3. * (lamb_r)) + (2. * (nu_r))))) * ((l ** 2) * ((W_r) * (rho_r))))) + aux96X
    aux98X = (-14. * ((np.exp((2. * ((lamb_r) + (nu_r))))) * ((l ** 2) * ((W_r) * (rho_r))))) + ((16. * ((np.exp(((lamb_r) + (2. * (nu_r))))) * ((l ** 2) * ((W_r) * (rho_r)))))+ aux97X)
    aux99X = (2. * ((np.exp(((3. * (lamb_r)) + (2. * (nu_r))))) * (l * ((W_r) * (rho_r))))) + ((-6. * ((np.exp((2. * (nu_r)))) * ((l ** 2) * ((W_r) * (rho_r))))) + aux98X)
    aux100X = (-10. * ((np.exp((2. * ((lamb_r) + (nu_r))))) * (l * ((W_r) * (rho_r))))) + ((14. * ((np.exp(((lamb_r) + (2. * (nu_r))))) * (l * ((W_r) * (rho_r))))) + aux99X)
    aux101X = (-512. * ((np.exp(((3. * (lamb_r)) + (nu_r)))) * ((pi ** 3.) * ((r ** 6.) *aux4X)))) + ((-6. * ((np.exp((2. * (nu_r)))) * (l * ((W_r) * (rho_r))))) + aux100X)
    aux102X = (-4. * ((np.exp(((2.5 * (lamb_r)) + (1.5 * (nu_r))))) * ((l ** 4.) * ((r ** 2)* (X_r))))) + (aux0X + (aux1X + ((-8. * aux2X) + ((-8. * aux3X) + aux101X))))
    aux103X = (4. * ((np.exp((1.5 * ((lamb_r) + (nu_r))))) * ((l ** 4.) * ((r ** 2) * (X_r))))) + aux102X
    aux104X = (-8. * ((np.exp(((2.5 * (lamb_r)) + (1.5 * (nu_r))))) * ((l ** 3.) * ((r ** 2)* (X_r))))) + aux103X
    aux105X = (8. * ((np.exp((1.5 * ((lamb_r) + (nu_r))))) * ((l ** 3.) * ((r ** 2) * (X_r))))) + aux104X
    aux106X = (-12. * ((np.exp((0.5 * ((lamb_r) + (3. * (nu_r)))))) * ((l ** 2) * ((r ** 2) *(X_r))))) + aux105X
    aux107X = (-8. * ((np.exp(((2.5 * (lamb_r)) + (1.5 * (nu_r))))) * ((l ** 2) * ((r ** 2) *(X_r))))) + aux106X
    aux108X = (20. * ((np.exp((1.5 * ((lamb_r) + (nu_r))))) * ((l ** 2) * ((r ** 2) * (X_r))))) + aux107X
    aux109X = (-12. * ((np.exp((0.5 * ((lamb_r) + (3. * (nu_r)))))) * (l * ((r ** 2) * (X_r))))) + aux108X
    aux110X = (-4. * ((np.exp(((2.5 * (lamb_r)) + (1.5 * (nu_r))))) * (l * ((r ** 2) * (X_r))))) + aux109X
    aux111X = ((np.exp(((lamb_r) + (nu_r)))) * (l * (1. + l))) - ((np.exp((lamb_r))) * ((r ** 2) * (omega ** 2)))
    aux112X = (r ** 2) * (((aux111X - ((r ** 2) * (omega ** 2))) - ((np.exp((nu_r))) * (l * (1. + l)))) * (p_r))
    aux113X = (r ** 4.) * ((((np.exp((nu_r))) * (l * (1. + l))) - ((r ** 2) * (omega ** 2))) * (((p_r) ** 2)))
    aux114X = (16. * ((np.exp(((lamb_r) + (nu_r)))) * (pi * aux112X))) + (64. * ((np.exp(((2. * (lamb_r)) + (nu_r)))) * ((pi ** 2) * aux113X)))
    aux115X = (-2. * ((np.exp(((lamb_r) + (nu_r)))) * ((1. + (l + (l ** 2))) * ((r ** 2) * (omega ** 2))))) + ((-4. * ((np.exp((lamb_r))) * ((r ** 4.) * (omega ** 4.)))) +aux114X)
    aux116X = (-2. * ((np.exp(((lamb_r) + (2. * (nu_r))))) * (l * (1. + l)))) + ((7. * ((np.exp((nu_r))) * ((r ** 2) * (omega ** 2)))) + aux115X)
    aux117X = ((np.exp((2. * (nu_r)))) * (l * (1. + l))) + (((np.exp((2. * ((lamb_r) + (nu_r))))) * (l * (1. + l))) + aux116X)
    aux118X = (aux117X - ((np.exp(((2. * (lamb_r)) + (nu_r)))) * ((r ** 2) * (omega ** 2)))) * ((p_r) + (rho_r))
    aux119X = ((16. * ((np.exp((1.5 * ((lamb_r) + (nu_r))))) * (l * ((r ** 2) * (X_r))))) +aux110X) - ((np.exp((0.5 * (lamb_r)))) * (l * ((1. + l) * ((r ** 2) * ((H1_r) * aux118X)))))
    aux120X = aux119X - ((np.exp(((3.5 * (lamb_r)) + (2. * (nu_r))))) * ((l ** 2) * ((r ** 2) * ((K_r) * (rho_r)))))
    aux121X = aux120X - ((np.exp(((3.5 * (lamb_r)) + (2. * (nu_r))))) * (l * ((r ** 2) * ((K_r) * (rho_r)))))
    aux122X = 0.125 * ((np.exp((0.5 * ((-nu_r) - (lamb_r))))) * ((r ** -5.) * ((omega ** -2.) * aux121X)))
    aux123X = ((np.exp((lamb_r))) * (1. + (l + (l ** 2)))) + (8. * ((np.exp((lamb_r))) * (pi * ((r ** 2) * (p_r)))))

    dX= aux122X / (-3. + aux123X)

    Xl = dX

    rhs_FD = [0]*4

    rhs_FD[0] = H1l
    rhs_FD[1] = Kl
    rhs_FD[2] = Wl
    rhs_FD[3] = Xl

    return np.array(rhs_FD)


#def integrate_origin(k0, w0, rmatch, bkg_func, l, omega, file_eos, r0, metodo): 

#    nu0 = bkg_func(ri_TOV)[1]
#    p0 = bkg_func(ri_TOV)[2]
#    rho0 = bkg_func(ri_TOV)[3]
#    gamma0 = bkg_func(ri_TOV)[4]

#    u0 = BC(r0, k0, w0, nu0, p0, rho0, gamma0, l, omega, file_eos)

#    N = int((rmatch - r0)/dr_pert)

#    sol_in = solve_ivp(rhs_FD, [r0, rmatch], u0, method=metodo, t_eval=np.linspace(r0, rmatch, N), args=(l, omega, bkg_func))

#    return sol_in

def integrate_origin(k0, w0, rmatch, bkg_func, l, omega, file_eos, r0): 
    
    nu0 = bkg_func(ri_TOV)[1]
    p0 = bkg_func(ri_TOV)[2]
    rho0 = bkg_func(ri_TOV)[3]
    gamma0 = bkg_func(ri_TOV)[4]   
    
    solver = ode(rhs_FD)
    solver.set_integrator('zvode',method='bdf')  #solver.set_integrator('dop853') 'dopri5' tentar sem essa linha    
    
    u0 = BC(r0, k0, w0, nu0, p0, rho0, gamma0, l, omega, file_eos).tolist()
    
    solver.set_initial_value(u0, r0)
    
    solver.set_f_params(l, omega, bkg_func)
    
    N = int((rmatch - r0)/dr_pert)
    
    r_values = np.linspace(r0, rmatch, N)
    
    u_values = []
    
    for r in r_values[1:]:
        u = solver.integrate(r)
        u_values.append(u)
        
    sol_in = np.array(u_values).tolist()
    
    return sol_in


def integrate_surface(h1r, kr, wr, rmatch, bkg_func, l, omega, R):    
        
    solver = ode(rhs_FD)
    solver.set_integrator('zvode',method='bdf')  #solver.set_integrator('dop853') 'dopri5' tentar sem essa linha    
    
    u0 = [h1r, kr, wr, 0]
    r0 = R-dr_pert
    
    solver.set_initial_value(u0, r0)
    
    solver.set_f_params(l, omega, bkg_func)
    
    N = int((R-dr_pert - rmatch)/dr_pert)
    
    r_values = np.linspace(R-dr_pert, rmatch, N)
    
    u_values = []
    
    for r in r_values[1:]:
        u = solver.integrate(r)
        u_values.append(u)
        
    sol_out = np.array(u_values)       
    
    return sol_out


def match_interior(rmatch, bkg_func, l, omega, file_eos, R, r0):
    
    integrate_origin_10 = integrate_origin(1, 0, rmatch, bkg_func, l, omega, file_eos, r0)
    integrate_origin_01 = integrate_origin(0, 1, rmatch, bkg_func, l, omega, file_eos, r0)
    integrate_surface_100 = integrate_surface(1, 0, 0, rmatch, bkg_func, l, omega, R)
    integrate_surface_010 = integrate_surface(0, 1, 0, rmatch, bkg_func, l, omega, R)
    integrate_surface_001 = integrate_surface(0, 0, 1, rmatch, bkg_func, l, omega, R)

    k0 = symbols("k0")
    w0 = symbols("w0")
    h1r = symbols("h1r")
    kr = symbols("kr")
    wr = symbols("wr")
      
    eq1 = Eq(k0 * integrate_origin_10[-1][0] + w0 * integrate_origin_01[-1][0] - h1r * integrate_surface_100[-1][0] - kr * integrate_surface_010[-1][0] - wr * integrate_surface_001[-1][0], 0)
    eq2 = Eq(k0 * integrate_origin_10[-1][1] + w0 * integrate_origin_01[-1][1] - h1r * integrate_surface_100[-1][1] - kr * integrate_surface_010[-1][1] - wr * integrate_surface_001[-1][1], 0)
    eq3 = Eq(k0 * integrate_origin_10[-1][2] + w0 * integrate_origin_01[-1][2] - h1r * integrate_surface_100[-1][2] - kr * integrate_surface_010[-1][2] - wr * integrate_surface_001[-1][2], 0)
    eq4 = Eq(k0 * integrate_origin_10[-1][3] + w0 * integrate_origin_01[-1][3] - h1r * integrate_surface_100[-1][3] - kr * integrate_surface_010[-1][3] - wr * integrate_surface_001[-1][3], 0)
    eq5 = Eq(k0, 1)

    sol = list(solve((eq1, eq2, eq3, eq4, eq5), (k0, w0, h1r, kr, wr)).values())

    return sol

def RW(h1r, kr, M, l, omega, R):

    aux0 = (R ** l) * ((-6. * (kr * M)) + ((2. * (h1r * (l * ((1. + l) * M)))) + ((kr - h1r) * (l * ((1. + l) * R)))))
    psi = (2. * aux0) / (((-1. + l) * (l * ((1. + l) * (2. + l)))) + ((0. + -12.j) * (M * omega)))

    aux0l = (1. + l) * ((-24. * (M ** 2)) + ((12. * (M * R)) + ((-1. + l) * (l * ((1. + l) * ((2. + l) * (R ** 2)))))))
    aux1l = 3. * (M * ((R ** 2) * ((l * ((1. + l) * (-4. + (l + (l ** 2))))) + (4. * ((R ** 2) * (omega ** 2))))))
    aux2l = ((-72. * (M ** 3.)) + ((12. * ((3. + (l + (l ** 2))) * ((M ** 2) * R))) + aux1l)) - ((-1. + l) * (l * ((1. + l) * ((2. + l) * (R ** 3.)))))
    aux3l = (h1r * (((2. * M) - R) * ((l * aux0l) + (-24. * (M * ((R ** 3.) * (omega ** 2))))))) + (-2. * (kr * aux2l))
    aux4l = ((0. + -1.j) * ((R ** (-1. + l)) * aux3l)) / (((0. + 1.j) * ((-1. + l) * (l * ((1. + l) * (2. + l))))) + (12. * (M * omega)))
    psil = (aux4l / ((6. * M) + ((-2. + (l + (l ** 2))) * R))) / ((-2. * M) + R)

    return psi, psil

def rhs_RW(r, u, l, omega, M):
    
    rhs_RW = np.array([u[1],
                       - 1/(r*(r - 2*M)) * (2 * M * u[1] + (r**3 * omega**2 / (r - 2* M) - l*(l+1) + 6*M/r) * u[0])])

    return rhs_RW    
    

def integrate_RW(phi0, phi0l, ri, rf, l, omega, M):
    
    u0 = [phi0, phi0l]
    
    N = int((rf - ri)/dr_pert)
    
    u = np.empty((N, 2))
    u[0] = u0
    r = np.linspace(ri, rf, N)
    
    for i in range(N-1):
        u[i+1] = rk4_step(r[i], u[i], dr_pert, rhs_RW, l, omega, M) 
    
    sol_intermediate = u
    
    return sol_intermediate[-1]
    


