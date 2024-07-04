import numpy as np
import astropy.units as u
from astropy.cosmology import Planck18, z_at_value
from scipy.special import gamma
import scipy.integrate as integrate
from tqdm import tqdm
import pandas as pd
from math import isnan
import scipy.interpolate as interpolate
from scipy.stats import gaussian_kde
from scipy.special import hyp2f1

# use astrophysical units throughout

# gravitation constant:
G_grav = 1/232.0

# speed of light:
c_light = 3.0e5

# Hubble time:
t_Hub = Planck18.age(0).value * 1e3

# Hubble constant:
H_0 = Planck18.H0.value / 1e6

# matter density parameter:
O_m0 = Planck18.Om0

# solar mass:
m_sun = 1.0

# solar radius:
r_sun = 2.26e-8

# redshfit of pop II star formation:
z_popII = 15.0

# age of pop II star formation:
t_popII = Planck18.age(z_popII).value * 1e3

# mass prior range:
Mcl0_min, Mcl0_max = 1e4, 1e7

# half-mass radius prior range:
rh0_min, rh0_max = 1e-2, 1e1

# cluster formation time prior range:
t0_min, t0_max = t_popII, t_Hub

# cluster formation redshift prior range:
zcl0_min, zcl0_max = 0.0, z_popII

# growth time prior range:
Dtg_min, Dtg_max = 100.0, t_Hub

# cluster formation redshift distribution parameters:
zcl0_mean, zcl0_sigma = 3.2, 1.5

# BH mass prior range:
mBH_min, mBH_max = 100.0, 1500.0

# critical escape velocity:
vesc0_crit = 200.0

# critical minimum BH mass:
MBH_crit = 100.0

# load age-redshift table (age in Myr):
age = np.load('./ages_redshifts_table_Planck18.npz')['age']
redshift = np.load('./ages_redshifts_table_Planck18.npz')['redshift']

def redshift_interp(t):
    """
    Interpolate age-redshift relation.
    
    Inputs:
    @in t: age of the Universe (in Myr)
    
    Outputs:
    @out z: redshift
    """
    
    z = np.interp(t, age, redshift)
    
    return z

def age_interp(z):
    """
    Interpolate redshift-age relation.
    
    Inputs:
    @in z: redshift
    
    Outputs:
    @out t: age of the Universe (in Myr)
    """
    
    t = np.interp(z, np.flip(redshift), np.flip(age))
    
    return t

def Hubble(z):
    """
    Hubble function.
    
    Inputs:
    @in z: redshift
    
    Output
    @out H: Hubble parameter
    """
    
    # Hubble parameter:
    H = H_0 * np.sqrt((1 + z)**3 * O_m0 + (1 - O_m0))
    
    return H
Hubble = np.vectorize(Hubble)

# initial average stellar mass in the cluster:
m_star0 = 0.6

# black hole seed mass:
M_BH0 = 10.0

# stellar-evolution timescale:
t_se = 2.0

# mean stellar mass evolution parameter:
nu = 0.07

# ejection parameter:
xi_e = 0.0074

# fraction of cluster's energy that can be conducted throughout the system in one half-mass relaxation time:
zeta = 0.0926

# multimass relaxation factor:
psi = 1.0

# Coulomb logarithm:
lnL = 10.0

# core collapse time normalized to initial half-mass relaxation time:
w_cc = 3.21

# black hole growth time normalized to initial half-mass relaxation time:
w_BH = 13.21

# potential energy form factor:
kappa = 0.4

# stellar cusp power law index:
g = 7/4.0

# tidal capture radius normalized to the tidal radius:
beta = 2.0

# tidal radius parameter:
eta = 0.844

# stellar radius:
R_star = r_sun

# fraction of stellar mass accreted in each tidal disruption event:
f_s = 0.50

def average_star_mass(t, N_star0=1e5, r_h0=1.2, m_star0=m_star0, M_BH0=M_BH0, t_se=t_se, nu=nu, xi_e=xi_e, zeta=zeta, psi=psi, lnL=lnL, w_cc=w_cc, w_BH=w_BH, kappa=kappa, g=g, beta=beta, eta=eta, R_star=R_star, f_s=f_s):
    
    if t < t_se:
        
        m_star = m_star0
        
    else:
        
        m_star = m_star0 * (t / t_se)**(-nu)
        
    return m_star
average_star_mass = np.vectorize(average_star_mass)

def star_number(t, N_star0=1e5, r_h0=1.2, m_star0=m_star0, M_BH0=M_BH0, t_se=t_se, nu=nu, xi_e=xi_e, zeta=zeta, psi=psi, lnL=lnL, w_cc=w_cc, w_BH=w_BH, kappa=kappa, g=g, beta=beta, eta=eta, R_star=R_star, f_s=f_s):
    
    # initial half-mass relaxation time:
    t_rh0 = 0.138 / psi / lnL * np.sqrt(N_star0 * r_h0**3 / G_grav / m_star0)
    
    # core collapse time:
    t_cc = w_cc * t_rh0
    
    if t < t_cc:
        
        N_star = N_star0
        
    else:
        
        x = 1 + (3*zeta - 7*xi_e) / 2 / t_rh0 * t_se**(2*nu) / t_cc**(3*nu) * (t**(nu+1) - t_cc**(nu+1)) / (nu+1)
        
        N_star = N_star0 * x**(2 * xi_e / (7*xi_e - 3*zeta))
        
    return N_star
star_number = np.vectorize(star_number)

def cluster_mass(t, N_star0=1e5, r_h0=1.2, m_star0=m_star0, M_BH0=M_BH0, t_se=t_se, nu=nu, xi_e=xi_e, zeta=zeta, psi=psi, lnL=lnL, w_cc=w_cc, w_BH=w_BH, kappa=kappa, g=g, beta=beta, eta=eta, R_star=R_star, f_s=f_s):
    
    # average star mass:
    m_star = average_star_mass(t, N_star0, r_h0, m_star0, M_BH0, t_se, nu, xi_e, zeta, psi, lnL, w_cc, w_BH, kappa, g, beta, eta, R_star, f_s)
    
    # star number:
    N_star = star_number(t, N_star0, r_h0, m_star0, M_BH0, t_se, nu, xi_e, zeta, psi, lnL, w_cc, w_BH, kappa, g, beta, eta, R_star, f_s)
    
    # black hole mass:
    M_BH = black_hole_mass(t, N_star0, r_h0, m_star0, M_BH0, t_se, nu, xi_e, zeta, psi, lnL, w_cc, w_BH, kappa, g, beta, eta, R_star, f_s)
    
    M_cl = m_star * N_star + M_BH
    
    return M_cl
cluster_mass = np.vectorize(cluster_mass)

def half_mass_radius(t, N_star0=1e5, r_h0=1.2, m_star0=m_star0, M_BH0=M_BH0, t_se=t_se, nu=nu, xi_e=xi_e, zeta=zeta, psi=psi, lnL=lnL, w_cc=w_cc, w_BH=w_BH, kappa=kappa, g=g, beta=beta, eta=eta, R_star=R_star, f_s=f_s):
    
    # initial half-mass relaxation time:
    t_rh0 = 0.138 / psi / lnL * np.sqrt(N_star0 * r_h0**3 / G_grav / m_star0)
    
    # core collapse time:
    t_cc = w_cc * t_rh0
    
    if t < t_se:
        
        r_h = r_h0
        
    elif t < t_cc:
        
        r_h = r_h0 * (t / t_se)**nu
        
    else:
        
        r_h = r_h0 * (t_cc**2 / t / t_se)**nu * (star_number(t, N_star0, r_h0, m_star0, M_BH0, t_se, nu, xi_e, zeta, psi, lnL, w_cc, w_BH, kappa, g, beta, eta, R_star, f_s) / N_star0)**(2 - zeta / xi_e)
        
    return r_h
half_mass_radius = np.vectorize(half_mass_radius)

def black_hole_mass(t, N_star0=1e5, r_h0=1.2, m_star0=m_star0, M_BH0=M_BH0, t_se=t_se, nu=nu, xi_e=xi_e, zeta=zeta, psi=psi, lnL=lnL, w_cc=w_cc, w_BH=w_BH, kappa=kappa, g=g, beta=beta, eta=eta, R_star=R_star, f_s=f_s):
    
    # initial half-mass relaxation time:
    t_rh0 = 0.138 / psi / lnL * np.sqrt(N_star0 * r_h0**3 / G_grav / m_star0)
    
    # core collapse time:
    t_cc = w_cc * t_rh0
    
    # BH growth start time:
    t_BH = w_BH * t_rh0
    
    if t < t_BH:
        
        M_BH = M_BH0
        
    else:
        
        c1 = np.sqrt(2 / np.pi) * (3 - g) / (2 - g) * gamma(g + 1) / gamma(g - 1/2)
        c2 = f_s * c1 * beta * eta**(2/3) * R_star * kappa**(5/2) * np.sqrt(G_grav) / 3**(5/2)
        c3 = 5/3 * c2 * m_star0**(13/6) * (N_star0 / r_h0)**(5/2) * t_se**(14*nu/3) / t_cc**(5*nu)
        c4 = (3*zeta - 7*xi_e) / 2 / t_rh0 * t_se**(2*nu) / t_cc**(3*nu) / (nu+1)
        A = nu/3
        B = c4 / (1 - c4 * t_cc**(nu+1))
        C = nu+1
        D = 5 * (zeta - xi_e) / (7*xi_e - 3*zeta)
        c5 = c3 * (1 - c4 * t_cc**(nu+1))**D / (A+1)
        
        M_BH = (M_BH0**(5/3) + c5 * (t**(A+1) * hyp2f1((A+1)/C, -D, (A+C+1)/C, -B*t**C) - t_BH**(A+1) * hyp2f1((A+1)/C, -D, (A+C+1)/C, -B*t_BH**C)))**(3/5)
        
    return M_BH
black_hole_mass = np.vectorize(black_hole_mass)

def black_hole_time(MBH, N_star0=1e5, r_h0=1.2, m_star0=m_star0, M_BH0=M_BH0, t_se=t_se, nu=nu, xi_e=xi_e, zeta=zeta, psi=psi, lnL=lnL, w_cc=w_cc, w_BH=w_BH, kappa=kappa, g=g, beta=beta, eta=eta, R_star=R_star, f_s=f_s):
    """
    Returns the time at which the black hole mass is MBH. Essentially inverts the MBH=f(t) function with the Newton-Raphson method.
    """
    
    # initial half-mass relaxation time:
    t_rh0 = 0.138 / psi / lnL * np.sqrt(N_star0 * r_h0**3 / G_grav / m_star0)
    
    # core collapse time:
    t_cc = w_cc * t_rh0
    
    # BH growth start time:
    t_BH = w_BH * t_rh0
    
    # mass of BH the cluster can make in a Hubble time:
    M_BHmax = black_hole_mass(t_Hub, N_star0, r_h0, m_star0, M_BH0, t_se, nu, xi_e, zeta, psi, lnL, w_cc, w_BH, kappa, g, beta, eta, R_star, f_s)
    
    # check if function has a root:
    if MBH <= M_BH0 or MBH > M_BHmax: # no real root exists
        
        return -1
        
    else: # a unique real root exists
        
        c1 = np.sqrt(2 / np.pi) * (3 - g) / (2 - g) * gamma(g + 1) / gamma(g - 1/2)
        c2 = f_s * c1 * beta * eta**(2/3) * R_star * kappa**(5/2) * np.sqrt(G_grav) / 3**(5/2)
        c3 = 5/3 * c2 * m_star0**(13/6) * (N_star0 / r_h0)**(5/2) * t_se**(14*nu/3) / t_cc**(5*nu)
        c4 = (3*zeta - 7*xi_e) / 2 / t_rh0 * t_se**(2*nu) / t_cc**(3*nu) / (nu+1)
        A = nu/3
        B = c4 / (1 - c4 * t_cc**(nu+1))
        C = nu+1
        D = 5 * (zeta - xi_e) / (7*xi_e - 3*zeta)
        c5 = c3 * (1 - c4 * t_cc**(nu+1))**D / (A+1)
        
        # Newton-Raphson root finding algorithm:
        
        t = t_BH # inital guess
        Dt = 1e100 # initial discrepancy
        tolerance = 1.0 # tolerance value
        
        while Dt > tolerance:
            
            f = (MBH**(5/3) - M_BH0**(5/3)) / c5 - (t**(A+1) * hyp2f1((A+1)/C, -D, (A+C+1)/C, -B*t**C) - t_BH**(A+1) * hyp2f1((A+1)/C, -D, (A+C+1)/C, -B*t_BH**C)) # function value
            
            dfdt = -(A+1) * t**A * hyp2f1((A+1)/C, -D, (A+C+1)/C, -B*t**C) - t**(A+C) * D*(A+1)*B*C/(A+C+1)*hyp2f1((A+C+1)/C, 1-D, (A+2*C+1)/C, -B*t**C) # derivative value
            
            t_new = t - f / dfdt # new proposal
            
            Dt = np.abs(t_new - t) # update discrepancy
            
            t = t_new # update guess
            
        return t
black_hole_time = np.vectorize(black_hole_time)

def delay_time(M_01, r_01, M_02, r_02, Dt_g1, Dt_g2, e, m_star0=m_star0, M_BH0=M_BH0, t_se=t_se, nu=nu, xi_e=xi_e, zeta=zeta, psi=psi, lnL=lnL, w_cc=w_cc, w_BH=w_BH, kappa=kappa, g=g, beta=beta, eta=eta, R_star=R_star, f_s=f_s):
    """
    Time from the moment of star cluster merger until the merger of the IMBH-IMBH binary.
    
    Returns:
    @out t_cc: core collapse time contribution
    @out t_hard: hardening time contribution
    @out t_gw: gravitational-wave time contribution
    @out t_delay: total time delay; t_delay = t_cc + t_hard + t_gw
    """
    
    # cluster masses just before cluster merger:
    M_1, M_2 = cluster_mass(Dt_g1, M_01 / m_star0, r_01, m_star0, M_BH0, t_se, nu, xi_e, zeta, psi, lnL, w_cc, w_BH, kappa, g, beta, eta, R_star, f_s), cluster_mass(Dt_g2, M_02 / m_star0, r_02, m_star0, M_BH0, t_se, nu, xi_e, zeta, psi, lnL, w_cc, w_BH, kappa, g, beta, eta, R_star, f_s)
    
    # half-mass radii just before cluster merger:
    r_1, r_2 = half_mass_radius(Dt_g1, M_01 / m_star0, r_01, m_star0, M_BH0, t_se, nu, xi_e, zeta, psi, lnL, w_cc, w_BH, kappa, g, beta, eta, R_star, f_s), half_mass_radius(Dt_g2, M_02 / m_star0, r_02, m_star0, M_BH0, t_se, nu, xi_e, zeta, psi, lnL, w_cc, w_BH, kappa, g, beta, eta, R_star, f_s)    
    
    # new cluster mass and half-mass radius assuming mass & energy conservation:
    M_0 = M_1 + M_2
    r_0 = M_0 / (M_1 / r_1 + M_2 / r_2)
    
    # initial half-mass relaxation time:
    t_rh0 = 0.138 / m_star0 / psi / lnL * np.sqrt(M_0 * r_0**3 / G_grav)
    
    # core collapse time:
    t_cc = w_cc * t_rh0
    
    # cluster energy:
    E_cl = - kappa / 2 * G_grav * M_0**2 / r_0
    
    # black hole masses just before cluster merger:
    M_BH1, M_BH2 = black_hole_mass(Dt_g1, M_01 / m_star0, r_01, m_star0, M_BH0, t_se, nu, xi_e, zeta, psi, lnL, w_cc, w_BH, kappa, g, beta, eta, R_star, f_s), black_hole_mass(Dt_g2, M_02 / m_star0, r_02, m_star0, M_BH0, t_se, nu, xi_e, zeta, psi, lnL, w_cc, w_BH, kappa, g, beta, eta, R_star, f_s)
    
    # hardening constant assuming balanced evolution:
    H = - 2 * zeta * E_cl / G_grav / M_BH1 / M_BH2 / t_rh0
    
    D = 64 / 5 * G_grav**3 * M_BH1 * M_BH2 * (M_BH1 + M_BH2) / c_light**5
    
    # eccentricity enhancement factor:
    Fe = (1 - e**2)**(-7/2) * (1 + 73 / 24 * e**2 + 37 / 96 * e**4)
    
    # gravitational wave semimajor axis:
    a_GW = (D / H * Fe)**(1 / 5)
    
    # hardening timescale:
    t_hard = 1 / H / a_GW
    
    # coalescence timescale for circular orbit:
    T_c = 5 * c_light**5 * a_GW**4 / (256 * G_grav**3 * M_BH1 * M_BH2 * (M_BH1 + M_BH2))
    
    # 1st order pN correction:
    S = 8**(1 - np.sqrt(1 - e)) * np.exp( 5 * G_grav * (M_BH1 + M_BH2) / c_light**2 / a_GW / (1 - e))
    
    # gravitational wave merger time:
    t_gw = T_c * (1 + 0.27 * e**10 + 0.33 * e**20 + 0.2 * e**1000) * (1 - e**2)**(7 / 2) * S
    
    # total delay time:
    t_delay = t_cc + t_hard + t_gw
    
    return t_cc, t_hard, t_gw, t_delay

def generate_IMBH_population(N_events=10**3, seed=453264573, return_cluster_samples=True, m_star0=m_star0, M_BH0=M_BH0, t_se=t_se, nu=nu, xi_e=xi_e, zeta=zeta, psi=psi, lnL=lnL, w_cc=w_cc, w_BH=w_BH, kappa=kappa, g=g, beta=beta, eta=eta, R_star=R_star, f_s=f_s, zcl0_mean=zcl0_mean):
    
    # initialize pseudo-random number generator:
    np.random.seed(seed)
    
    # evaporation coefficient:
    w_ev = w_cc - 2 / (3 * zeta - 7 * xi_e)
    
    m1, m2, zm = [], [], []
    
    Mcl0_1, Mcl0_2, rh0_1, rh0_2, Dtg_1, Dtg_2, Dtd, zcl0_1, zcl0_2 = [], [], [], [], [], [], [], [], []
    
    # generate events:
    for i in tqdm(range(N_events)):
        
        # sample cluster masses:
        Mcl0_1_i, Mcl0_2_i = ((Mcl0_min)**(-1) + ((Mcl0_max)**(-1) - (Mcl0_min)**(-1)) * np.random.rand(2))**(-1)
        
        # sample cluster half-mass radii:
        rh0_1_i, rh0_2_i = 10**np.random.uniform(np.log10(rh0_min), np.log10(rh0_max), size=2)
        
        # cluster initial-half-mass relaxation times:
        trh0_1_i, trh0_2_i = 0.138 / psi / lnL / m_star0 * np.sqrt(Mcl0_1_i * rh0_1_i**3 / G_grav), 0.138 / psi / lnL / m_star0 * np.sqrt(Mcl0_2_i * rh0_2_i**3 / G_grav)
        
        # cluster evaporation times:
        tev_1_i, tev_2_i = Dtg_min if w_ev < 0 else w_ev * trh0_1_i, np.inf if w_ev < 0 else w_ev * trh0_2_i
        
        # cluster core collapse times:
        tcc_1_i, tcc_2_i = w_cc * trh0_1_i, w_cc * trh0_2_i
        
        # cluster BH times:
        tBH_1_i, tBH_2_i = w_BH * trh0_1_i, w_BH * trh0_2_i
        
        # sample cluster formation redshifts:
        zcl0_1_i, zcl0_2_i = np.random.normal(loc=zcl0_mean, scale=zcl0_sigma, size=2)
        
        # sample first growth time:
        Dtg_1_i = (tBH_1_i**(1/2) + (Dtg_max**(1/2) - tBH_1_i**(1/2))*np.random.rand())**(2) if w_ev < 0 else (tBH_1_i**(1/2) + (tev_1_i**(1/2) - tBH_1_i**(1/2))*np.random.rand())**(2)
        
        # compute second growth time:
        Dtg_2_i = Dtg_1_i + age_interp(zcl0_1_i) - age_interp(zcl0_2_i)
        Dtg_2_i = min(Dtg_2_i, 0.999 * tev_2_i) # make sure Dtg_2_i < tev_2_i (strict)
        # compute merger delay time:
        Dtd_i = delay_time(Mcl0_1_i, rh0_1_i, Mcl0_2_i, rh0_2_i, Dtg_1_i, Dtg_2_i, np.sqrt(np.random.rand()), m_star0, M_BH0, t_se, nu, xi_e, zeta, psi, lnL, w_cc, w_BH, kappa, g, beta, eta, R_star, f_s)[3]
        
        # compute total delay times:
        Dt_1_i, Dt_2_i = Dtg_1_i + Dtd_i, Dtg_2_i + Dtd_i
        
        # calculate BH masses:
        m1_i = black_hole_mass(Dtg_1_i, Mcl0_1_i / m_star0, rh0_1_i, m_star0, M_BH0, t_se, nu, xi_e, zeta, psi, lnL, w_cc, w_BH, kappa, g, beta, eta, R_star, f_s)
        m2_i = black_hole_mass(Dtg_2_i, Mcl0_2_i / m_star0, rh0_2_i, m_star0, M_BH0, t_se, nu, xi_e, zeta, psi, lnL, w_cc, w_BH, kappa, g, beta, eta, R_star, f_s)
        
        while max(age_interp(zcl0_1_i) + Dt_1_i, age_interp(zcl0_2_i) + Dt_2_i) > t_Hub or min(m1_i, m2_i) < mBH_min or max(m1_i, m2_i) > mBH_max or Dtg_2_i < tBH_2_i or Dtg_2_i < 0: #or Dtg_2_i > tev_2_i:
            
            # sample cluster masses:
            Mcl0_1_i, Mcl0_2_i = ((Mcl0_min)**(-1) + ((Mcl0_max)**(-1) - (Mcl0_min)**(-1)) * np.random.rand(2))**(-1)
            
            # sample cluster half-mass radii:
            rh0_1_i, rh0_2_i = 10**np.random.uniform(np.log10(rh0_min), np.log10(rh0_max), size=2)
            
            # cluster initial-half-mass relaxation times:
            trh0_1_i, trh0_2_i = 0.138 / psi / lnL / m_star0 * np.sqrt(Mcl0_1_i * rh0_1_i**3 / G_grav), 0.138 / psi / lnL / m_star0 * np.sqrt(Mcl0_2_i * rh0_2_i**3 / G_grav)
            
            # cluster evaporation times:
            tev_1_i, tev_2_i = Dtg_min if w_ev < 0 else w_ev * trh0_1_i, np.inf if w_ev < 0 else w_ev * trh0_2_i
            
            # cluster core collapse times:
            tcc_1_i, tcc_2_i = w_cc * trh0_1_i, w_cc * trh0_2_i
            
            # cluster BH times:
            tBH_1_i, tBH_2_i = w_BH * trh0_1_i, w_BH * trh0_2_i
            
            # sample cluster formation redshifts:
            zcl0_1_i, zcl0_2_i = np.random.normal(loc=zcl0_mean, scale=zcl0_sigma, size=2)
            
            # sample first growth time:
            Dtg_1_i = (tBH_1_i**(1/2) + (Dtg_max**(1/2) - tBH_1_i**(1/2))*np.random.rand())**(2) if w_ev < 0 else (tBH_1_i**(1/2) + (tev_1_i**(1/2) - tBH_1_i**(1/2))*np.random.rand())**(2)
            
            # compute second growth time:
            Dtg_2_i = Dtg_1_i + age_interp(zcl0_1_i) - age_interp(zcl0_2_i)
            Dtg_2_i = min(Dtg_2_i, 0.999 * tev_2_i) # make sure Dtg_2_i < tev_2_i (strict)
            # compute merger delay time:
            Dtd_i = delay_time(Mcl0_1_i, rh0_1_i, Mcl0_2_i, rh0_2_i, Dtg_1_i, Dtg_2_i, np.sqrt(np.random.rand()), m_star0, M_BH0, t_se, nu, xi_e, zeta, psi, lnL, w_cc, w_BH, kappa, g, beta, eta, R_star, f_s)[3]
            
            # compute total delay times:
            Dt_1_i, Dt_2_i = Dtg_1_i + Dtd_i, Dtg_2_i + Dtd_i
            
            # calculate BH masses:
            m1_i = black_hole_mass(Dtg_1_i, Mcl0_1_i / m_star0, rh0_1_i, m_star0, M_BH0, t_se, nu, xi_e, zeta, psi, lnL, w_cc, w_BH, kappa, g, beta, eta, R_star, f_s)
            m2_i = black_hole_mass(Dtg_2_i, Mcl0_2_i / m_star0, rh0_2_i, m_star0, M_BH0, t_se, nu, xi_e, zeta, psi, lnL, w_cc, w_BH, kappa, g, beta, eta, R_star, f_s)
            
        # compute BBH merger redshift:
        zm_i = redshift_interp(Dt_1_i + age_interp(zcl0_1_i))
        
        m1.append(m1_i)
        m2.append(m2_i)
        zm.append(zm_i)
        
        Mcl0_1.append(Mcl0_1_i)
        Mcl0_2.append(Mcl0_2_i)
        rh0_1.append(rh0_1_i)
        rh0_2.append(rh0_2_i)
        Dtg_1.append(Dtg_1_i)
        Dtg_2.append(Dtg_2_i)
        Dtd.append(Dtd_i)
        zcl0_1.append(zcl0_1_i)
        zcl0_2.append(zcl0_2_i)
        
    m1 = np.array(m1)
    m2 = np.array(m2)
    zm = np.array(zm)
    
    Mcl0_1 = np.array(Mcl0_1)
    Mcl0_2 = np.array(Mcl0_2)
    rh0_1 = np.array(rh0_1)
    rh0_2 = np.array(rh0_2)
    Dtg_1 = np.array(Dtg_1)
    Dtg_2 = np.array(Dtg_2)
    Dtd = np.array(Dtd)
    zcl0_1 = np.array(zcl0_1)
    zcl0_2  = np.array(zcl0_2)
    
    IMBH_population = {'m1': m1, 'm2': m2, 'zm': zm, 'Mcl0_1': Mcl0_1, 'rh0_1': rh0_1, 'zcl0_1': zcl0_1, 'Dtg_1': Dtg_1, 'Mcl0_2': Mcl0_2, 'rh0_2': rh0_2, 'zcl0_2': zcl0_2, 'Dtg_2': Dtg_2, 'Dtd': Dtd} if return_cluster_samples else {'m1': m1, 'm2': m2, 'zm': zm}
    
    return IMBH_population

def sample_single_cluster_point(m1, m2, zm, seed=2456823, m_star0=m_star0, M_BH0=M_BH0, t_se=t_se, nu=nu, xi_e=xi_e, zeta=zeta, psi=psi, lnL=lnL, w_cc=w_cc, w_BH=w_BH, kappa=kappa, g=g, beta=beta, eta=eta, R_star=R_star, f_s=f_s):
    
    # instantiate the random number generator:
    np.random.seed(seed)
    
    # age of Universe at the redshift of merger:
    tm = age_interp(zm)
    
    # evaporation coefficient:
    w_ev = w_cc - 2 / (3 * zeta - 7 * xi_e)
    
    # sample first cluster:
    Mcl0_1_i = 10**np.random.uniform(np.log10(Mcl0_min), np.log10(Mcl0_max)) # cluster mass
    rh0_min = max((G_grav / Mcl0_1_i * (m_star0 * lnL * t_se / 0.138 / w_cc)**2)**(1/3), 4 * kappa * G_grav * Mcl0_1_i / vesc0_crit**2) # minimum initial half-mass radius
    rh0_max = min((G_grav / Mcl0_1_i * (m_star0 * lnL * tm / 0.138 / w_BH)**2)**(1/3), 1e-11 * Mcl0_1_i**(2.5)) # maximum initial half-mass radius
    rh0_1_i = 10**np.random.uniform(np.log10(rh0_min), np.log10(rh0_max)) # initial half-mass radius
    vesc0_1_i = 2 * np.sqrt(kappa * G_grav * Mcl0_1_i / rh0_1_i) # initial escape velocity
    trh0_1_i = 0.138 * np.sqrt(Mcl0_1_i * rh0_1_i**3 / G_grav) / m_star0 / lnL / psi # initial half-mass relaxation time
    tBH_1_i = w_BH * trh0_1_i # BH evaporation time
    tcc_1_i = w_cc * trh0_1_i # core collapse time
    # the BH cluster can make in a Hubble time:
    MBH1_max_i = black_hole_mass(t_Hub if w_ev < 0 else w_ev * trh0_1_i, Mcl0_1_i / m_star0, rh0_1_i, m_star0, M_BH0, t_se, nu, xi_e, zeta, psi, lnL, w_cc, w_BH, kappa, g, beta, eta, R_star, f_s)
    # first BH growth time:
    Dtg_1_i = black_hole_time(m1, Mcl0_1_i / m_star0, rh0_1_i, m_star0, M_BH0, t_se, nu, xi_e, zeta, psi, lnL, w_cc, w_BH, kappa, g, beta, eta, R_star, f_s)
    
    # sample second cluster:
    Mcl0_2_i = 10**np.random.uniform(np.log10(Mcl0_min), np.log10(Mcl0_max)) # cluster mass
    rh0_min = max((G_grav / Mcl0_2_i * (m_star0 * lnL * t_se / 0.138 / w_cc)**2)**(1/3), 4 * kappa * G_grav * Mcl0_2_i / vesc0_crit**2) # minimum initial half-mass radius
    rh0_max = min((G_grav / Mcl0_2_i * (m_star0 * lnL * tm / 0.138 / w_BH)**2)**(1/3), 1e-11 * Mcl0_2_i**(2.5)) # maximum initial half-mass radius
    rh0_2_i = 10**np.random.uniform(np.log10(rh0_min), np.log10(rh0_max)) # initial half-mass radius
    vesc0_2_i = 2 * np.sqrt(kappa * G_grav * Mcl0_2_i / rh0_2_i) # initial escape velocity
    trh0_2_i = 0.138 * np.sqrt(Mcl0_2_i * rh0_2_i**3 / G_grav) / m_star0 / lnL / psi # relaxation time
    tBH_2_i = w_BH * trh0_2_i # BH evaporation time
    tcc_2_i = w_cc * trh0_2_i # core collapse time
    # the BH cluster can make in a Hubble time:
    MBH2_max_i = black_hole_mass(t_Hub if w_ev < 0 else w_ev * trh0_2_i, Mcl0_2_i / m_star0, rh0_2_i, m_star0, M_BH0, t_se, nu, xi_e, zeta, psi, lnL, w_cc, w_BH, kappa, g, beta, eta, R_star, f_s)
    # second BH growth time:
    Dtg_2_i = black_hole_time(m2, Mcl0_2_i / m_star0, rh0_2_i, m_star0, M_BH0, t_se, nu, xi_e, zeta, psi, lnL, w_cc, w_BH, kappa, g, beta, eta, R_star, f_s)
    
    # delay time:
    Dtd_i = delay_time(Mcl0_1_i, rh0_1_i, Mcl0_2_i, rh0_2_i, Dtg_1_i, Dtg_2_i, np.sqrt(np.random.rand()), m_star0, M_BH0, t_se, nu, xi_e, zeta, psi, lnL, w_cc, w_BH, kappa, g, beta, eta, R_star, f_s)[3]
    
    # total times:
    Dt_1_i = Dtg_1_i + Dtd_i
    Dt_2_i = Dtg_2_i + Dtd_i
    
    # cluster formation times:
    tcl0_1_i = tm - Dt_1_i
    tcl0_2_i = tm - Dt_2_i
    
    while tBH_1_i > t_Hub or vesc0_1_i > vesc0_crit or MBH1_max_i < MBH_crit or Dtg_1_i < 0 \
       or tcc_1_i < t_se or isnan(float(MBH1_max_i))==True or tcl0_1_i < t_popII \
       or tBH_2_i > t_Hub or vesc0_2_i > vesc0_crit or MBH2_max_i < MBH_crit or Dtg_2_i < 0 \
       or tcc_2_i < t_se or isnan(float(MBH2_max_i))==True or tcl0_2_i < t_popII:
        
        # sample first cluster:
        Mcl0_1_i = 10**np.random.uniform(np.log10(Mcl0_min), np.log10(Mcl0_max)) # cluster mass
        rh0_min = max((G_grav / Mcl0_1_i * (m_star0 * lnL * t_se / 0.138 / w_cc)**2)**(1/3), 4 * kappa * G_grav * Mcl0_1_i / vesc0_crit**2) # minimum initial half-mass radius
        rh0_max = min((G_grav / Mcl0_1_i * (m_star0 * lnL * tm / 0.138 / w_BH)**2)**(1/3), 1e-11 * Mcl0_1_i**(2.5)) # maximum initial half-mass radius
        rh0_1_i = 10**np.random.uniform(np.log10(rh0_min), np.log10(rh0_max)) # initial half-mass radius
        vesc0_1_i = 2 * np.sqrt(kappa * G_grav * Mcl0_1_i / rh0_1_i) # initial escape velocity
        trh0_1_i = 0.138 * np.sqrt(Mcl0_1_i * rh0_1_i**3 / G_grav) / m_star0 / lnL / psi # initial half-mass relaxation time
        tBH_1_i = w_BH * trh0_1_i # BH evaporation time
        tcc_1_i = w_cc * trh0_1_i # core collapse time
        # the BH cluster can make in a Hubble time:
        MBH1_max_i = black_hole_mass(t_Hub if w_ev < 0 else w_ev * trh0_1_i, Mcl0_1_i / m_star0, rh0_1_i, m_star0, M_BH0, t_se, nu, xi_e, zeta, psi, lnL, w_cc, w_BH, kappa, g, beta, eta, R_star, f_s)
        # first BH growth time:
        Dtg_1_i = black_hole_time(m1, Mcl0_1_i / m_star0, rh0_1_i, m_star0, M_BH0, t_se, nu, xi_e, zeta, psi, lnL, w_cc, w_BH, kappa, g, beta, eta, R_star, f_s)
        
        # sample second cluster:
        Mcl0_2_i = 10**np.random.uniform(np.log10(Mcl0_min), np.log10(Mcl0_max)) # cluster mass
        rh0_min = max((G_grav / Mcl0_2_i * (m_star0 * lnL * t_se / 0.138 / w_cc)**2)**(1/3), 4 * kappa * G_grav * Mcl0_2_i / vesc0_crit**2) # minimum initial half-mass radius
        rh0_max = min((G_grav / Mcl0_2_i * (m_star0 * lnL * tm / 0.138 / w_BH)**2)**(1/3), 1e-11 * Mcl0_2_i**(2.5)) # maximum initial half-mass radius
        rh0_2_i = 10**np.random.uniform(np.log10(rh0_min), np.log10(rh0_max)) # initial half-mass radius
        vesc0_2_i = 2 * np.sqrt(kappa * G_grav * Mcl0_2_i / rh0_2_i) # initial escape velocity
        trh0_2_i = 0.138 * np.sqrt(Mcl0_2_i * rh0_2_i**3 / G_grav) / m_star0 / lnL / psi # relaxation time
        tBH_2_i = w_BH * trh0_2_i # BH evaporation time
        tcc_2_i = w_cc * trh0_2_i # core collapse time
        # the BH cluster can make in a Hubble time:
        MBH2_max_i = black_hole_mass(t_Hub if w_ev < 0 else w_ev * trh0_2_i, Mcl0_2_i / m_star0, rh0_2_i, m_star0, M_BH0, t_se, nu, xi_e, zeta, psi, lnL, w_cc, w_BH, kappa, g, beta, eta, R_star, f_s)
        # second BH growth time:
        Dtg_2_i = black_hole_time(m2, Mcl0_2_i / m_star0, rh0_2_i, m_star0, M_BH0, t_se, nu, xi_e, zeta, psi, lnL, w_cc, w_BH, kappa, g, beta, eta, R_star, f_s)
        
        # delay time:
        Dtd_i = delay_time(Mcl0_1_i, rh0_1_i, Mcl0_2_i, rh0_2_i, Dtg_1_i, Dtg_2_i, np.sqrt(np.random.rand()), m_star0, M_BH0, t_se, nu, xi_e, zeta, psi, lnL, w_cc, w_BH, kappa, g, beta, eta, R_star, f_s)[3]
        
        # total times:
        Dt_1_i = Dtg_1_i + Dtd_i
        Dt_2_i = Dtg_2_i + Dtd_i
        
        # cluster formation times:
        tcl0_1_i = tm - Dt_1_i
        tcl0_2_i = tm - Dt_2_i
        
    # cluster formation redshifts:
    zcl0_1_i = redshift_interp(tcl0_1_i)
    zcl0_2_i = redshift_interp(tcl0_2_i)
    
    return np.log10(Mcl0_1_i), np.log10(rh0_1_i), zcl0_1_i, Dtg_1_i, \
           np.log10(Mcl0_2_i), np.log10(rh0_2_i), zcl0_2_i, Dtg_2_i, Dtd_i

def sample_cluster_space(m1, m2, zm, N_samples=1000, seed=4573272, with_tqdm=False, m_star0=m_star0, M_BH0=M_BH0, t_se=t_se, nu=nu, xi_e=xi_e, zeta=zeta, psi=psi, lnL=lnL, w_cc=w_cc, w_BH=w_BH, kappa=kappa, g=g, beta=beta, eta=eta, R_star=R_star, f_s=f_s):
    
    # instantiate the random number generator:
    np.random.seed(seed)
    
    if with_tqdm: # show progress bar
        
        cluster_samples_array = np.array([sample_single_cluster_point(m1, m2, zm, np.random.randint(999999999), m_star0, M_BH0, t_se, nu, xi_e, zeta, psi, lnL, w_cc, w_BH, kappa, g, beta, eta, R_star, f_s) for i in tqdm(range(N_samples))]).T
        
    else: # do not show progress bar
        
        cluster_samples_array = np.array([sample_single_cluster_point(m1, m2, zm, np.random.randint(999999999), m_star0, M_BH0, t_se, nu, xi_e, zeta, psi, lnL, w_cc, w_BH, kappa, g, beta, eta, R_star, f_s) for i in range(N_samples)]).T
    
    cluster_samples_dictionary = {
        'log_Mcl0_1': cluster_samples_array[0], 
        'log_rh0_1':cluster_samples_array[1], 
        'zcl0_1': cluster_samples_array[2], 
        'Dtg_1': cluster_samples_array[3], 
        'log_Mcl0_2': cluster_samples_array[4], 
        'log_rh0_2': cluster_samples_array[5], 
        'zcl0_2': cluster_samples_array[6], 
        'Dtg_2': cluster_samples_array[7], 
        'Dtd': cluster_samples_array[8]
    }
    
    return cluster_samples_dictionary

def generate_cluster_posterior_samples(source_posterior_samples, N_samples_per_source_sample=1000, N_samples=10**4, seed=43452352, with_tqdm=False, m_star0=m_star0, M_BH0=M_BH0, t_se=t_se, nu=nu, xi_e=xi_e, zeta=zeta, psi=psi, lnL=lnL, w_cc=w_cc, w_BH=w_BH, kappa=kappa, g=g, beta=beta, eta=eta, R_star=R_star, f_s=f_s, zcl0_mean=zcl0_mean):
    
    # instantiate the random number generator:
    np.random.seed(seed)
    
    # number of source posterior samples:
    N_source_posterior_samples = source_posterior_samples['mass_1_source'].size
    
    log_Mcl0_1, log_rh0_1, zcl0_1, Dtg_1, log_Mcl0_2, log_rh0_2, zcl0_2, Dtg_2, Dtd = \
    [], [], [], [], [], [], [], [], []
    
    if with_tqdm: # show progress bar
        
        for i in tqdm(range(N_source_posterior_samples)):
            
            m1 = source_posterior_samples['mass_1_source'][i] # first BH mass sample
            m2 = source_posterior_samples['mass_2_source'][i] # second BH mass sample
            zm = redshift_interp(source_posterior_samples['merger_time'][i] * 1000) # merger redshift sample
            
            # generate cluster samples for particular source sample:
            cluster_posterior_samples_i = sample_cluster_space(m1, m2, zm, N_samples_per_source_sample, np.random.randint(999999999), False, m_star0, M_BH0, t_se, nu, xi_e, zeta, psi, lnL, w_cc, w_BH, kappa, g, beta, eta, R_star, f_s)
            
            log_Mcl0_1.append(cluster_posterior_samples_i['log_Mcl0_1'])
            log_rh0_1.append(cluster_posterior_samples_i['log_rh0_1'])
            zcl0_1.append(cluster_posterior_samples_i['zcl0_1'])
            Dtg_1.append(cluster_posterior_samples_i['Dtg_1'])
            log_Mcl0_2.append(cluster_posterior_samples_i['log_Mcl0_2'])
            log_rh0_2.append(cluster_posterior_samples_i['log_rh0_2'])
            zcl0_2.append(cluster_posterior_samples_i['zcl0_2'])
            Dtg_2.append(cluster_posterior_samples_i['Dtg_2'])
            Dtd.append(cluster_posterior_samples_i['Dtd'])
            
    else: # do not show progress bar
        
        for i in range(N_source_posterior_samples):
            
            m1_i = source_posterior_samples['mass_1_source'][i] # first BH mass sample
            m2_i = source_posterior_samples['mass_2_source'][i] # second BH mass sample
            zm_i = redshift_interp(source_posterior_samples['merger_time'][i] * 1000) # merger redshift sample
            
            # generate cluster samples for particular source sample:
            cluster_posterior_samples_i = sample_cluster_space(m1_i, m2_i, zm_i, N_samples_per_source_sample, np.random.randint(999999999), False, m_star0, M_BH0, t_se, nu, xi_e, zeta, psi, lnL, w_cc, w_BH, kappa, g, beta, eta, R_star, f_s)
            
            log_Mcl0_1.append(cluster_posterior_samples_i['log_Mcl0_1'])
            log_rh0_1.append(cluster_posterior_samples_i['log_rh0_1'])
            zcl0_1.append(cluster_posterior_samples_i['zcl0_1'])
            Dtg_1.append(cluster_posterior_samples_i['Dtg_1'])
            log_Mcl0_2.append(cluster_posterior_samples_i['log_Mcl0_2'])
            log_rh0_2.append(cluster_posterior_samples_i['log_rh0_2'])
            zcl0_2.append(cluster_posterior_samples_i['zcl0_2'])
            Dtg_2.append(cluster_posterior_samples_i['Dtg_2'])
            Dtd.append(cluster_posterior_samples_i['Dtd'])
            
    log_Mcl0_1 = np.concatenate(np.array(log_Mcl0_1))
    log_rh0_1 = np.concatenate(np.array(log_rh0_1))
    zcl0_1 = np.concatenate(np.array(zcl0_1))
    Dtg_1 = np.concatenate(np.array(Dtg_1))
    log_Mcl0_2 = np.concatenate(np.array(log_Mcl0_2))
    log_rh0_2 = np.concatenate(np.array(log_rh0_2))
    zcl0_2 = np.concatenate(np.array(zcl0_2))
    Dtg_2 = np.concatenate(np.array(Dtg_2))
    Dtd = np.concatenate(np.array(Dtd))
    
    # remove NaN elements:
    condition = (~np.isnan(Dtg_1)) & (~np.isnan(Dtg_2))
    log_Mcl0_1 = log_Mcl0_1[condition]
    log_rh0_1 = log_rh0_1[condition]
    zcl0_1 = zcl0_1[condition]
    Dtg_1 = Dtg_1[condition]
    log_Mcl0_2 = log_Mcl0_2[condition]
    log_rh0_2 = log_rh0_2[condition]
    zcl0_2 = zcl0_2[condition]
    Dtg_2 = Dtg_2[condition]
    Dtd = Dtd[condition]
    
    # downsample with prior:
    # ------------------------------------------------------------------------------------------------------------------------
    # weight probability (the astro prior on cluster parameters):
    p = np.exp(-((zcl0_1 - zcl0_mean)**2 + (zcl0_2 - zcl0_mean)**2) / 2 / zcl0_sigma**2) / 10**(2 * log_Mcl0_1) / 10**(2 * log_Mcl0_2) * 10**(log_Mcl0_1) * 10**(log_Mcl0_2) * (10**(log_Mcl0_1) / np.sqrt(Dtg_1) + 10**(log_Mcl0_2) / np.sqrt(Dtg_2))
    
    # normalized weight factor:
    p = p / np.sum(p)
    
    # downsample sample indices:
    indices = np.random.choice(np.arange(Dtd.size), replace=True, size=N_samples, p=p)
    
    # get samples:
    log_Mcl0_1 = log_Mcl0_1[indices]
    log_rh0_1 = log_rh0_1[indices]
    zcl0_1 = zcl0_1[indices]
    Dtg_1 = Dtg_1[indices]
    log_Mcl0_2 = log_Mcl0_2[indices]
    log_rh0_2 = log_rh0_2[indices]
    zcl0_2 = zcl0_2[indices]
    Dtg_2 = Dtg_2[indices]
    Dtd = Dtd[indices]
    # ------------------------------------------------------------------------------------------------------------------------
    
    cluster_posterior_samples = {
        'log_Mcl0_1': log_Mcl0_1, 
        'log_rh0_1': log_rh0_1, 
        'zcl0_1': zcl0_1, 
        'Dtg_1': Dtg_1, 
        'log_Mcl0_2': log_Mcl0_2, 
        'log_rh0_2': log_rh0_2, 
        'zcl0_2': zcl0_2, 
        'Dtg_2': Dtg_2, 
        'Dtd': Dtd
    }
    
    return cluster_posterior_samples

# end of file.
