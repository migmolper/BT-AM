#!/usr/local/bin/python3.10

import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.interpolate import CubicSpline
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
# for Palatino and other serif fonts use:
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})
# It's also possible to use the reduced notation by directly setting font.family:
plt.rcParams.update({
  "text.usetex": True,
  "font.family": "Helvetica"
})

Order = 44

p_monic = special.hermite(Order, monic=True)
x_k, w_k = np.polynomial.hermite.hermgauss(Order)
print(x_k,w_k)
