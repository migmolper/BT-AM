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

reader = open('Mg_H.adp.alloy.txt')
try:
    for i in range(0,3): line = reader.readline()

    # 
    line = reader.readline()
    print(line.split())

    # 
    line = reader.readline()
    info = line.split()
    N_rho = int(info[0])
    d_rho = float(info[1])
    N_r = int(info[2])
    d_r = float(info[3])
    cutoff_r = float(info[4])
    r_ij = np.linspace(0.0,cutoff_r,N_r)

    # Mg atomistic data
    atomistic_info = reader.readline().split() # 
    Mg_atomic_number = int(atomistic_info[0])
    Mg_mass = float(atomistic_info[1])
    Mg_lattice_constant = float(atomistic_info[2])
    Mg_lattice_type = atomistic_info[3]

    # Mg embedding function and energy density function
    Mg_data_embedding_function = np.zeros(N_rho)
    for i in range(0,N_rho):
        Mg_data_embedding_function[i] = float(reader.readline().split()[0])
    Mg_energy_density_function_data = np.zeros(N_r)
    for i in range(0,N_r):
        Mg_energy_density_function_data[i] = float(reader.readline().split()[0])

    # H atomistic data
    atomistic_info = reader.readline().split() #
    H_atomic_number = int(atomistic_info[0])
    H_mass = float(atomistic_info[1])
    H_lattice_constant = float(atomistic_info[2])
    H_lattice_type = atomistic_info[3]

    # H embedding function and energy density function
    H_embedding_function_data = np.zeros(N_rho)
    for i in range(0,N_rho):
        H_embedding_function_data[i] = float(reader.readline().split()[0])
    H_energy_density_function_data = np.zeros(N_r)
    for i in range(0,N_r):
        H_energy_density_function_data[i] = float(reader.readline().split()[0])

    # Mg-Mg pair potential
    MgMg_pair_potential_data = np.zeros(N_r)
    for i in range(0,N_r):
        MgMg_pair_potential_data[i] = float(reader.readline().split()[0])

    # Mg-H pair potential
    MgH_pair_potential_data = np.zeros(N_r)
    for i in range(0,N_r):
        MgH_pair_potential_data[i] = float(reader.readline().split()[0])

    # H-H pair potential
    HH_pair_potential_data = np.zeros(N_r)
    for i in range(0,N_r):
        HH_pair_potential_data[i] = float(reader.readline().split()[0])

    # Mg-Mg u function (dipole)
    MgMg_dipole_function_u_data = np.zeros(N_r)
    for i in range(0,N_r):
        MgMg_dipole_function_u_data[i] = float(reader.readline().split()[0])

    # Mg-H u function (dipole)
    MgH_dipole_function_u_data = np.zeros(N_r)
    for i in range(0,N_r):
        MgH_dipole_function_u_data[i] = float(reader.readline().split()[0])

    # H-H u function (dipole)
    HH_dipole_function_u_data = np.zeros(N_r)
    for i in range(0,N_r):
        HH_dipole_function_u_data[i] = float(reader.readline().split()[0])

    # Mg-Mg u function (quadrupole)
    MgMg_quadrupole_function_w_data = np.zeros(N_r)
    for i in range(0,N_r):
        MgMg_quadrupole_function_w_data[i] = float(reader.readline().split()[0])

    # Mg-H u function (quadrupole)
    MgH_quadrupole_function_w_data = np.zeros(N_r)
    for i in range(0,N_r):
        MgH_quadrupole_function_w_data[i] = float(reader.readline().split()[0])

    # H-H u function (quadrupole)
    HH_quadrupole_function_w_data = np.zeros(N_r)
    for i in range(0,N_r):
        HH_quadrupole_function_w_data[i] = float(reader.readline().split()[0])

finally:
    reader.close()

##########

cs_dipole_MgMg = CubicSpline(r_ij, MgMg_dipole_function_u_data)
xs = np.arange(0.01, 6.5, 0.01)
plt.title(r'Dipole function $u$') 
plt.plot(xs, cs_dipole_MgMg(xs), label=r'spline Mg-Mg')
plt.show()

Order = 1

p_monic = special.hermite(Order, monic=True)
x_k, w_k = np.polynomial.hermite.hermgauss(Order+1)
print(x_k,w_k)

x = np.linspace(-3, 3, 400)
plt.plot(x, p_monic(x)*np.exp(-x**2))
plt.plot(x, p_monic(x))
plt.title("Hermite polynomial of degree {}".format(Order))
plt.xlabel("x")
plt.ylabel("H_2(x)")
plt.show()

integral_Hn = 0
integral_u = 0
for i in range(0,Order+1):
    integral_Hn += w_k[i]*p_monic(x_k[i])
    integral_u +=  w_k[i]*cs_dipole_MgMg(x_k[i])

print("Gaussian integral with Hermite quad [-inf,inf] of the H_{} = {}".format(Order,integral_Hn))
print("Gaussian integral with Hermite quad [-inf,inf] of the dipole-u = {}".format(integral_u))


integral_Hn = integrate.quad(lambda x: p_monic(x)*np.exp(-x**2), -np.inf,np.inf)
print(integral_Hn)

integral_u = integrate.quad(lambda x: cs_dipole_MgMg(x)*np.exp(-x**2), -np.inf,np.inf)
print(integral_u)

print("Gaussian integral [-inf,inf] of the H_{} = {}".format(Order,integral_Hn))
print("Gaussian integral [-inf,inf] of the dipole-u = {}".format(integral_u))