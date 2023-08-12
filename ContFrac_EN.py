from Eq_freqD import *

def alpha(n, a, M, l):
    return n*(n+1)*(a -2*M)

def beta(n, a, M, l, omega):
    return n*(6*M*n -2*a*n -2*1j*omega*a**2)

def gamma(n, a, M, l):
    return 6*M -6*M*n*(n-1) -a*l*(l+1) +a*n*(n-1)

def delta(n, a, M, l):
        return 2*M*(n-2)*n -6*M

def match_int_ext(omega_lst, l, n, rmatch, bkg_func, file_eos, R, M, r0):

    omega = omega_lst[0]+1j*omega_lst[1]

    ctes = match_interior(rmatch, bkg_func, l, omega, file_eos, R, r0)
    k0, h0, h1r, kr, wr = [complex(complex_num.as_real_imag()[0], complex_num.as_real_imag()[1]) for complex_num in ctes]
    phi, phil = RW(h1r, kr, M, l, omega, R)

    if M/R < 0.25:
        a = R
    else:
        a = 5*M
        uinterm = integrate_RW(phi, phil, R, a, l, omega, M)
        psi = uinterm[0]
        psil = uinterm[1]

    list_alpha = [-1, 2*(a-2*M)]
    list_beta = [R*(phil/phi + (1j*R*omega)/(-2*M+R)), 6*M -2*a*(1+1j*a*omega)]
    list_gamma = [0,-a*l*(l+1)+6*M]
    list_delta = [0,0]

    for i in range(2, n+1):
        list_alpha.append(alpha(i, a, M, l))
        list_beta.append(beta(i, a, M, l, omega))
        list_gamma.append(gamma(i, a, M, l))
        list_delta.append(delta(i, a, M, l))

    def αhat(n, a, M, l):
        if n<2:
            return list_alpha[n]
        else:
            return alpha(n, a, M, l)

    def γhat(n, a, M, l, omega):
        if n<2:
            return list_gamma[n]
        else:
            return gamma(n, a, M, l) - (βhat(n-1, a, M, l, omega)*delta(n, a, M, l))/γhat(n-1, a, M, l, omega)

    def βhat(n, a, M, l, omega):
        if n<2:
            return list_beta[n]
        else:
            return beta(n, a, M, l, omega) - (αhat(n-1, a, M, l)*delta(n, a, M, l))/γhat(n-1, a, M, l, omega)

    cf = 0

    for i in reversed(range(1,n+1)):
        cf = cf + βhat(i, a, M, l, omega)
        cf = (-γhat(i, a, M, l, omega) * αhat(i-1, a, M, l)) / cf

    res = βhat(0, a, M, l, omega) + cf
    abs_res = np.abs(res)

    return [res.real, res.imag]