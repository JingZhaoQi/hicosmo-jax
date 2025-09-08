"""
Physical and astronomical constants used in HiCosmo.

All constants are in SI units unless otherwise specified.
"""

# Speed of light
c_km_s = 299792.458  # km/s
c_m_s = 299792458.0  # m/s

# Gravitational constant
G_SI = 6.67430e-11  # m^3 kg^-1 s^-2
G_cgs = 6.67430e-8  # cm^3 g^-1 s^-2

# Planck constant
h = 6.62607015e-34  # J·s
hbar = 1.054571817e-34  # J·s

# Boltzmann constant
k_B = 1.380649e-23  # J/K

# Stefan-Boltzmann constant
sigma_sb = 5.670374419e-8  # W m^-2 K^-4

# Radiation density constant
a_rad = 7.5657e-16  # J m^-3 K^-4

# Critical density factor
# rho_crit = 3H^2/(8πG) 
rho_crit_h2 = 2.77536627e11  # h^2 M_sun/Mpc^3

# CMB temperature today
T_cmb = 2.7255  # K

# Solar mass
M_sun = 1.98847e30  # kg

# Parsec
pc = 3.0857e16  # m
kpc = 3.0857e19  # m
Mpc = 3.0857e22  # m
Gpc = 3.0857e25  # m

# Year
yr = 365.25 * 24 * 3600  # s
Gyr = 1e9 * yr  # s

# Hubble time
t_H = 1.0 / (100 * 3.24078e-20)  # s for H0=100 km/s/Mpc
t_H_Gyr = t_H / Gyr  # Gyr for H0=100 km/s/Mpc

# Hubble distance
d_H = c_km_s / 100  # Mpc for H0=100 km/s/Mpc

# Radiation density parameters
# Photons
Omega_gamma_h2 = 2.47e-5  # Photon density parameter * h^2

# Neutrinos (3 species, standard model)
N_eff = 3.046  # Effective number of neutrino species
Omega_nu_h2 = 1.68e-5  # Neutrino density parameter * h^2 (massless)

# Total radiation
Omega_r_h2 = Omega_gamma_h2 * (1 + 0.2271 * N_eff)  # Total radiation * h^2

# Conversion factors
eV_to_J = 1.602176634e-19  # eV to Joules
eV_to_kg = 1.782661921e-36  # eV/c^2 to kg

# Thomson scattering
sigma_T = 6.6524587321e-29  # m^2

# Fine structure constant
alpha = 1.0 / 137.035999084

# Electron mass
m_e = 9.1093837015e-31  # kg
m_e_eV = 0.51099895000e6  # eV/c^2

# Proton mass
m_p = 1.67262192369e-27  # kg
m_p_eV = 938.27208816e6  # eV/c^2

# Neutron mass
m_n = 1.67492749804e-27  # kg
m_n_eV = 939.56542052e6  # eV/c^2

# Primordial helium mass fraction (BBN prediction)
Y_p = 0.2453  # Primordial helium-4 mass fraction

# Reionization parameters (Planck 2018)
z_reion = 7.67  # Reionization redshift
tau_reion = 0.054  # Optical depth to reionization

# BAO scale
r_s_fid = 147.09  # Mpc, fiducial sound horizon at drag epoch

# Default cosmological parameters (Planck 2018 + BAO)
DEFAULT_COSMO = {
    'H0': 67.66,       # Hubble constant in km/s/Mpc
    'Omega_m': 0.3111,  # Total matter density
    'Omega_b': 0.04897, # Baryon density
    'Omega_c': 0.2621,  # Cold dark matter density
    'sigma8': 0.8102,   # Matter fluctuation amplitude
    'n_s': 0.9665,      # Scalar spectral index
    'tau': 0.0561,      # Optical depth to reionization
}

# Unit conversions for convenience
def Mpc_to_m(d_Mpc):
    """Convert distance from Mpc to meters."""
    return d_Mpc * Mpc

def m_to_Mpc(d_m):
    """Convert distance from meters to Mpc."""
    return d_m / Mpc

def H0_to_h(H0):
    """Convert H0 in km/s/Mpc to dimensionless h."""
    return H0 / 100.0

def h_to_H0(h):
    """Convert dimensionless h to H0 in km/s/Mpc."""
    return h * 100.0

def z_to_a(z):
    """Convert redshift to scale factor."""
    return 1.0 / (1.0 + z)

def a_to_z(a):
    """Convert scale factor to redshift."""
    return 1.0 / a - 1.0