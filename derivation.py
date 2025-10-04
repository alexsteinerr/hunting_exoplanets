
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math
import matplotlib.pyplot as plt

# ----------------- Physical constants -----------------
G = 6.67430e-11       # m^3 kg^-1 s^-2
c = 2.99792458e8      # m/s
pi = math.pi

# ----------------- Stellar / orbital parameters -----------------
M_star = 1.989e30                 # stellar mass [kg] (example ~Sun)
P      = 3.0 * 86400.0            # orbital period [s]  (example: 3 days)

# ----------------- Limb-darkening (quadratic) -----------------
# I(mu_LD)/I0 = 1 - u1(1 - mu_LD) - u2(1 - mu_LD)^2
u1, u2 = 0.3, 0.2                 # example coefficients

# ----------------- Inclination evolution i(t) ------------------
# i(t) = i0 + Δi * sin(2π t / P_i)
i0_deg     = 89.5                 # mean inclination [deg]
Delta_i_deg= 0.10                 # oscillation amplitude [deg]
P_i        = 100.0 * 86400.0      # precession/variation period [s]
i0      = np.deg2rad(i0_deg)
Delta_i = np.deg2rad(Delta_i_deg)

# ----------------- Transparency factor μ -----------------------
# μ in [0, μ_max]; blockage scales with (1 - μ/μ_max).
mu_max = 1.0
mu_const = 0.0                    # set >0 for partial transparency; 0 → opaque

def mu_of_t(tt):
    """Planet transparency factor μ(t). For now, constant; customize if needed."""
    return np.full_like(np.asarray(tt, dtype=float), mu_const, dtype=float)

# ----------------- Transit window parameters -------------------
R_s = 9.3681570600e+08        # star radius [m]
R_p = 7.7900948857e+07        # planet radius [m]
tau = 9.7292529297e+03        # total transit duration [s]
t0  = 2.8424736907e+04        # mid-transit [s]

# ----------------- Photon bending controls ---------------------
ENABLE_BENDING = True
# (Exact multiplicative cosine as per user's formula)
# Note: this can flip sign; set ENABLE_BENDING=False if undesired.
# For small modulation instead, multiply I_rel by (1 + eps*(cos(theta)-1)).

# ----------------- Derived ratios & scales ---------------------
k = R_p / R_s                                  # radius ratio
a = (G * M_star * P**2 / (4.0 * pi**2))**(1.0/3.0)   # semi-major axis [m]

# Time grid around one transit (±tau as in the user's code)
t = np.linspace(t0 - tau, t0 + tau, 4001)

# ----------------- Functions -----------------
def i_of_t(tt):
    """Time-dependent inclination [rad]."""
    return i0 + Delta_i * np.sin(2.0*pi*tt / P_i)

def b_of_t(tt):
    """Impact parameter in stellar radii: b(t) = a cos i(t) / R_s."""
    return (a * np.cos(i_of_t(tt))) / R_s

def overlap_normalized(zval, k):
    """
    Overlap function λ(z; k): fraction of stellar flux blocked by the planet,
    normalized by stellar disk area (no limb darkening).
    """
    zval = np.asarray(zval)
    out = np.zeros_like(zval)

    mask0 = (zval >= 1.0 + k)   # no overlap
    mask1 = (zval <= 1.0 - k)   # full overlap
    mask2 = (~mask0) & (~mask1) # partial

    out[mask0] = 0.0
    out[mask1] = k**2

    zz = zval[mask2]

    def clamp(u):
        return np.clip(u, -1.0, 1.0)

    # Geometry terms
    term1 = np.arccos(clamp((zz**2 + 1 - k**2) / (2*zz)))
    term2 = np.arccos(clamp((zz**2 + k**2 - 1) / (2*zz*k)))
    term3 = 0.5 * np.sqrt(np.clip((-zz + 1 + k)*(zz + 1 - k)*(zz - 1 + k)*(zz + 1 + k), 0, None))
    A = term1 + k**2 * term2 - term3
    out[mask2] = A / math.pi
    return out

# ----------------- Geometry & kinematics -----------------------
# Use b(t0) to set chord and v_R (constant vR during the window).
b0 = float(b_of_t(t0))
chord0 = 2.0 * math.sqrt(max(0.0, (1.0 + k)**2 - b0**2))   # length in R_s
vR = chord0 / tau                                          # speed across disk [R_s / s]

# Planet position along the chord (in R_s) and projected separation z(t)
x = vR * (t - t0)                  # in R_s
b_t = b_of_t(t)                    # in R_s
z = np.sqrt(b_t**2 + x**2)         # in R_s

# ----------------- Overlap and limb darkening ------------------
lam_geom = overlap_normalized(z, k)   # geometric blocked fraction

# Quadratic limb darkening at planet center:
mu_LD = np.sqrt(np.clip(1.0 - z**2, 0.0, 1.0))  # classic LD μ (cos θ on stellar disk)
I_rel_local = 1.0 - u1*(1.0 - mu_LD) - u2*(1.0 - mu_LD)**2
I_rel_local = np.clip(I_rel_local, 0.0, 1.0)

# Limb-darkened blocked fraction (approx.)
lam_LD = lam_geom * I_rel_local

# ----------------- Apply transparency factor μ ------------------
mu_t = mu_of_t(t)
mu_t = np.clip(mu_t, 0.0, mu_max)
effective_block = lam_LD * (1.0 - mu_t / mu_max)

# Relative intensity without bending
I_rel = 1.0 - effective_block

# ----------------- Photon bending (updated phase) ---------------
# θ(t) = 4 G M_star / (c^2 z(t) R_s)
# Guard z=0 at exact center with a tiny epsilon.
eps = 1e-12
theta = (4.0 * G * M_star) / (c**2 * np.maximum(z, eps) * R_s)

if ENABLE_BENDING:
    I_rel_bent = I_rel * np.cos(theta)
else:
    I_rel_bent = I_rel

# ----------------- Diagnostics -----------------
dt = np.diff(t)
mid_drop = 0.5 * (effective_block[:-1] + effective_block[1:])
drop_integral = np.sum(mid_drop * dt)

print("k = Rp/Rs =", k)
print("b(t0) =", b0)
print("Chord(t0) [R_s] =", chord0)
print("v_R [R_s/s] =", vR)
print("μ_const =", mu_const, "  (μ_max =", mu_max, ")")
print("∫ΔI(t) dt =", drop_integral, "s")

# ----------------- Plots -----------------
plt.figure(figsize=(8,5))
plt.plot(t, effective_block)
plt.fill_between(t, effective_block, 0.0, alpha=0.3, label="Effective deficit ΔI(t) with μ")
plt.axvline(t0 - 0.5*tau, linestyle="--")
plt.axvline(t0 + 0.5*tau, linestyle="--")
plt.xlabel("Time (s)")
plt.ylabel("Deficit ΔI(t)")
plt.title("Transit Deficit with μ-transparency, LD, and i(t)")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8,5))
plt.plot(t, I_rel_bent)
plt.axhline(1.0, linestyle="--")
plt.xlabel("Time (s)")
plt.ylabel("Relative Intensity")
plt.title("Observed Intensity: μ-transparency + LD + i(t) " + ("+ photon bending" if ENABLE_BENDING else ""))
plt.grid(True)
plt.show()
