import numpy as np
from scipy.constants import hbar,c,e,h,mu_0,electron_volt,electron_mass,epsilon_0
from scipy.interpolate import CubicSpline
from Presets import presets
from StopPowerCalculator import *

#########################################################################################
#                                   Material Class                                      #
#                                                                                       #
#  Estimates the stopping power experienced by a magnetic monopole propagating through  #
#  a given material, based on its elemental composition and local density.              #
#  Required for defining the materials used by the primitives composing a Simulation-   #
#  Volume object.                                                                       #
#########################################################################################

class Material:
    # Initialization of the class. If using air, please use the preset
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
        self.interpolator=None

    # The getDensity module is called through one of the object classes, which take the 
    # position in the appropriate coordinate system. In order to avoid issues with the 
    # coordinates, it is ill advised to call this method directly.
    
    def getDensity(self,*kwargs):
        if callable(self.density):
            return self.density(*kwargs)
        else:
            return self.density
    
    def getElecDensity(self,position):
        elec_density=0

        for element in self.composition.keys():
            Z,mol_mass,exc_energy=ATOMS[element]
            elec_density+=self.getDensity(*position)*self.composition[element]*Z/mol_mass*N_A

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
            atom_density+=self.getDensity(*position)*N_A*self.composition[element]/mol_mass

        return atom_density
    
    def getDenCorrection(self,position,beta):
        I=self.getExcEnergy()*10**9
        x=self.getDenParam(beta)

        if self.preset=="Air":
            R=self.getDensity(*position)/rho_0

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
        
    # Wrapper function for the actual stopping power estimation, 
    # responsible for distributing calculations to the appropriate
    # mathematical models

    def getStopPower(self,position,beta,charge):
        if beta<1e-4:
            return giacomelli(self.getAtDensity(position))
        
        # The Bartlett estimate for the monopole stopping power in air 
        # should not be used for other gasses. Neither should the Ahlen-Kinoshita 
        # formula, as it has been idealized for non-conducting condensates.

        elif self.preset=="Air":
            return bartlett(beta,charge,self.getDensity(*position))
        else:   
            if beta<0.01:
                return ahlen_kinoshita(self.getElecDensity(position),beta,charge)
            elif beta>=0.1:
                return bethe(self.getElecDensity(position),beta,charge,self.getExcEnergy(),self.getDenCorrection(position,beta))
            else:
                ak=np.linspace(1e-4,1e-2,1000)
                bb=np.linspace(1e-1,np.nextafter(1.0, 0.0),100)
                betas=[]
                SP=[]
                for B in ak:
                    betas.append(B)
                    SP.append(ahlen_kinoshita(self.getElecDensity(position),B,charge))
                for B in bb:
                    betas.append(B)
                    SP.append(bethe(self.getElecDensity(position),B,charge,self.getExcEnergy(),self.getDenCorrection(position,B)))
                return np.interp(beta,betas,SP)         

