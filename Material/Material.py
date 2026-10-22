import numpy as np
from scipy.constants import hbar,c,e,h,mu_0,electron_volt,electron_mass,epsilon_0
from scipy.interpolate import CubicSpline
from Presets import presets

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
}

g_D=h/(mu_0*e)
b=2*(3*np.pi**2)**(1/3)*bohr
B=CubicSpline([1,2,3,6],[0.248,0.672,1.022,1.685])

def ahlen_kinoshita(elec_density,beta,charge):
    a=2*np.pi*(charge*g_D*mu_0*c/(4*np.pi))**2*e**2/(hbar*c*(3*np.pi**2)**(1/3))
    return a*elec_density**(2/3)*(np.log(b*elec_density**(1/3))-1/2)*beta/GeV

def bethe(elec_density,beta,charge,exc_energy,den_correction):
    if abs(charge)==1:
        K=0.406
    else:
        K=0.346
    gamma=1/np.sqrt(1-beta**2)
    A=np.pi*hbar**2/electron_mass*elec_density*charge**2
    log_argument=2*(electron_mass*c**2/GeV)*beta**2*gamma**2/exc_energy
    return A*(np.log(log_argument)+K/2-den_correction/2-1/2-B(abs(charge)))/GeV

def giacomelli(atom_density):
    return atom_density*hbar**2/(electron_mass*GeV)

def bartlett(beta,charge,density,position=None):
    gamma=1/np.sqrt(1-beta**2)

    if beta>=0.03:
        S=1.55*charge**2*(np.log(gamma*beta)+4)
        if callable(density):
            return 0.1*S*density(position)
        else:
            return 0.1*S*density
        
    else:
        S=860*charge**2*beta**2
        if callable(density):
            return 0.1*S*density(position)
        else:
            return 0.1*S*density
        
class Material:
    def __init__(self,composition,density):
        if isinstance(composition,str) and composition in presets:
            self.preset=composition
            self.composition=presets[composition]
        elif isinstance(composition,str) and composition not in presets:
            raise KeyError("Preset not recognized. Add it to presets or provide composition")
        else:
            self.preset=None
            self.composition=composition
        self.density=density

    def getElecDensity(self,position):
        elec_density=0

        for element in self.composition.keys():
            Z,mol_mass,exc_energy=ATOMS[element]
            if callable(self.density):
                elec_density+=self.density(position)*self.composition[element]*Z/mol_mass*N_A
            else:
                elec_density+=self.density*self.composition[element]*Z/mol_mass*N_A

        return elec_density

    def getPlasEnergy(self,position):
        return hbar*np.sqrt(self.getElecDensity(position)*e**2/(electron_mass*epsilon_0))/(10**9*electron_volt)
    
    def getDenParam(self,beta):
        gamma=1/np.sqrt(1-beta**2)
        return np.log10(beta*gamma)
    
    def getExcEnergy(self):
        upper=0
        lower=0

        for element in self.composition.keys():
            Z,mol_mass,exc_energy=ATOMS[element]
            upper+=self.composition[element]*Z/mol_mass*np.log(exc_energy)
            lower+=self.composition[element]*Z/mol_mass

        return np.exp(upper/lower)

    def getSternParam(self,position):
        return 1+2*np.log(self.getExcEnergy()/self.getPlasEnergy(position))

    def getAtDensity(self,position):
        atom_density=0
        for element in self.composition.keys():
            Z,mol_mass,exc_energy=ATOMS[element]
            if callable(self.density):
                atom_density+=self.density(position)*N_A*self.composition[element]/mol_mass
            else:
                atom_density+=self.density*N_A*self.composition[element]/mol_mass
        return atom_density
    
    def getDenCorrection(self,position,beta):
        I=self.getExcEnergy()*10**9
        x=self.getDenParam(beta)
        r,theta,phi=position

        if self.preset=="Air":
            if callable(self.density):
                R=self.density(r,theta,phi)/rho_0
            else:
                R=self.density/rho_0
            x0=x0_air-1/2*np.log10(R)
            x1=x1_air-1/2*np.log10(R)
            C=C_air-np.log(R)
            k=k_air
        else:
            C=self.getSternParam(position)
            k=3
            x1=2 if I<100 else 3

            if I<100:
                x0=0.2 if C<3.681 else 0.326*C-1
            else:
                x0=0.2 if C<5.215 else 0.326*C-1.5

        a=(C-2*np.log(10)*x0)/(x1-x0)**k

        if x<x0:
            return 0
        elif x>=x0 and x<x1:
            return 2*np.log(10)*x-C+a*(x1-x)**k
        else:
            return 2*np.log(10)*x-C

    def getStopPower(self,position,beta,charge):
        if beta<1e-4:
            return giacomelli(self.getAtDensity(position))
        elif self.preset=="Air":
            return bartlett(beta,charge,self.density,position)
        else:   
            if beta<0.01:
                return ahlen_kinoshita(self.getElecDensity(position),beta,charge)
            elif beta>0.1:
                return bethe(self.getElecDensity(position),beta,charge,self.getExcEnergy(),self.getDenCorrection(position,beta))
            else:
                ak=np.linspace(1e-4,1e-2,10000)
                bb=np.linspace(1e-1,1,1000)
                betas=[]
                SP=[]
                for B in ak:
                    betas.append(B)
                    SP.append(ahlen_kinoshita(self.getElecDensity(position),B,charge))
                for B in bb:
                    betas.append(B)
                    SP.append(bethe(self.getElecDensity(position),B,charge,self.getExcEnergy(),self.getDenCorrection(position,B)))
                return np.interp(beta,betas,SP)         

