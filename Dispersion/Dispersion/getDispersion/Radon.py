import numpy as np

def Radon(uxt, dx, dt, vmin, vmax, dv, fmin, fmax, df):
    """
    Radon transform
    """
    nt, ng = uxt.shape

    lf = int(round(fmin / df)) + 1
    nf = int(round(fmax / df)) + 1

    np_length = int((vmax - vmin) / dv + 1)
    ccn = int(1 / df / dt)

    d = np.fft.fft(uxt, ccn, axis=0).T

    x = np.arange(dx, dx * (ng+1), dx)
    pp = 1./np.arange(vmin, vmax + dv, dv)
    ll0 = 1j * 2 * np.pi * df * (pp[:, None] * x)
    mm = np.zeros((np_length, nf), dtype=complex)

    for luoj in range(lf, nf):
        l = np.exp(ll0 * (luoj - 1))
        mm[:, luoj] = np.dot(l, d[:, luoj])

    ml = np.abs(mm)
    ml = ml[:, lf:nf]

    return ml