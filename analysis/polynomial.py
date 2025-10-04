import numpy as np
from numpy.polynomial import Chebyshev, Polynomial
from config.settings import *

def fit_chebyshev_polynomial(p_dense, y_nn_dense, dip_center):
    domain = [dip_center - HALF_WINDOW, dip_center + HALF_WINDOW]
    cheb = Chebyshev.fit(p_dense, y_nn_dense, deg=CHEB_DEGREE, domain=domain)
    poly = cheb.convert(kind=Polynomial)
    
    # Format equation string
    terms = []
    for k, a_k in enumerate(poly.coef):
        coeff = f"{a_k:.10e}"
        if k == 0:   terms.append(f"{coeff}")
        elif k == 1: terms.append(f"{coeff}·p")
        else:        terms.append(f"{coeff}·p^{k}")
    equation_str = "y(p) = " + " + ".join(terms)
    
    return cheb, poly, equation_str