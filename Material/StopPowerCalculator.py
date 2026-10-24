import numpy as np
from scipy.constants import hbar,c,e,h,mu_0,electron_volt,electron_mass,epsilon_0
from scipy.interpolate import CubicSpline

N_A = 6.02214076e23  # mol^-1
bohr=5.29e-11
GeV=1e9*electron_volt
x0_air=1.7418
x1_air=4.2759
k_air=3.3994
C_air=10.5961
rho_0=1.205

ATOMS = {
    # Symbol: (Z, molar_mass_kg_per_mol, mean_excitation_energy_GeV)
    "H":  (1,  1.008e-3, 19.2e-9),
    "C":  (6, 12.011e-3, 81.0e-9),
    "O":  (8, 15.999e-3, 95.0e-9),
    "Na": (11, 22.990e-3, 149.0e-9),
    "Mg": (12, 24.305e-3, 156.0e-9),
    "Al": (13, 26.982e-3, 166.0e-9),
    "Si": (14, 28.085e-3, 173.0e-9),
    "P":  (15, 30.974e-3, 173.0e-9),
    "S":  (16, 32.06e-3, 180.0e-9),
    "K":  (19, 39.098e-3, 190.0e-9),
    "Ca": (20, 40.078e-3, 191.0e-9),
    "Ti": (22, 47.867e-3, 233.0e-9),
    "Mn": (25, 54.938e-3, 272.0e-9),
    "Fe": (26, 55.845e-3, 286.0e-9),
    "Ni": (28, 58.693e-3, 311.0e-9),
    "N":  (7,  14.007e-3, 82.0e-9),
    "Ar": (18, 39.948e-3, 188.0e-9),
    "Cl": (17, 35.453e-3, 174.0e-9),
    "Br": (35, 79.904e-3, 343.0e-9),
}
g_D=h/(mu_0*e)
b=2*(3*np.pi**2)**(1/3)*bohr
B=CubicSpline([1,2,3,6],[0.248,0.672,1.022,1.685])

# The Ahlen-Kinoshita equation is valid for non-conducting condensates at monopole
# speeds between 1e-4 and 1e-2. Extrapolating for other media or speeds will produce 
# unreliable results

def ahlen_kinoshita(elec_density,beta,charge):
    a=2*np.pi*(charge*g_D*mu_0*c/(4*np.pi))**2*e**2/(hbar*c*(3*np.pi**2)**(1/3))
    return a*elec_density**(2/3)*(np.log(b*elec_density**(1/3))-1/2)*beta/GeV

# The Bethe-Bloch corrected form of the Ahlen equation is valid for
# faster monopoles, with speeds between 0.1 and 1. Extrapolating for speeds will produce 
# unreliable results

def bethe(elec_density,beta,charge,exc_energy,den_correction):
    if abs(charge)==1:
        K=0.406
    else:
        K=0.346
    gamma=1/np.sqrt(1-beta**2)
    A=np.pi*hbar**2/electron_mass*elec_density*charge**2
    log_argument=2*(electron_mass*c**2/GeV)*beta**2*gamma**2/exc_energy
    return A*(np.log(log_argument)+K/2-den_correction/2-1/2-B(abs(charge)))/GeV

# The Giacomelli function represents a rough estimate given during a lecture in the 
# Lake Louise Winter Institute lecture for the stopping power experinced by a monopole with 
# speed beta below 1e-4, and represents loses by atomic collisions.
# Extrapolating for speeds will produce unreliable results

def giacomelli(atom_density):
    return atom_density*hbar**2/(electron_mass*GeV)

# The Bartlett implementation uses the air stopping power estimates by Bartlett for 
# ionization losses. As the monopole will no longer ionize the medium when at speeds 
# below 1e-4, it is necessary that we transition to the Giacomelli estimate when getting
# to this speed.
def bartlett(beta,charge,density):
    gamma=1/np.sqrt(1-beta**2)

    if beta>=0.03:
        S=1.55*charge**2*(np.log(gamma*beta)+4)
        return 0.1*S*density
        
    else:
        S=860*charge**2*beta**2
        return 0.1*S*density
