def rk3_step(t, u, h, f, *args):
    k1 = h * f(t      , u              , *args)
    k2 = h * f(t + h/2, u + 1/2 * k1   , *args)
    k3 = h * f(t + h  , u - k1 + 2 * k2, *args)
    return u + 1/6*(k1 + 4*k2 + k3)


def rk4_step(r, u, h, f, *args):
    k1 = h * f(r, u, *args)
    k2 = h * f(r + h/2, u + k1/2, *args)
    k3 = h * f(r + h/2, u + k2/2, *args)
    k4 = h * f(r + h, u + k3, *args)
    u_novo = u + (k1 + 2.0 * k2 + 2.0 * k3 + k4)/6.0
    return u_novo
