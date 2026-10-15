import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

# ODEs and miscellaneous calcuations for the propagation.

def get_stopping_power(density,stop_power,beta):
    if callable(stop_power):
        return 0.1*density*stop_power(beta)
    else:
        return 0.1*density*stop_power

def pos_ode(direction):
    drds =  direction
    return drds

def mom_ode(mag_field,stop_power,direction,charge,beta):
    dpds= (20.5*charge*mag_field - stop_power*direction)/beta
    return dpds

def mag_work_ode(mag_field,direction, charge):
    work = 20.5*charge*np.dot(mag_field,direction)
    return work 

def stop_work_ode(stop_power):
    work = stop_power
    return work

class Monopole:
    def __init__(self, beta, position, direction, mass,charge, mag_work=0, stop_work=0):

        # The monopole properties are set here. This will be changed as the monopole is propagated.

        self.lorentzFactor = 1/np.sqrt(1-beta**2)
        self.beta = beta
        self.totalEnergy = self.lorentzFactor * mass
        self.kineticEnergy = self.totalEnergy - mass 
        self.position = position
        self.direction = direction
        self.mass = mass
        self.charge = charge
        self.magnetic_work = mag_work
        self.stopping_work = stop_work
        self.momentum = np.sqrt(self.totalEnergy**2-self.mass**2)*direction

    # Making the propagation a method of the monopole class makes it so the same monopole, with the
    # same initial properties, can be propagated for different conditions. 
    # This should make things considerably easier for the different tests. 

    def propagate(self,magnetic_field, stop_power,density,limit):
        
        def ode(s, y):
            position = y[:3]
            direction = y[3:6] /np.linalg.norm(y[3:6])
            p= np.linalg.norm(y[3:6])
            beta= p/np.sqrt(p**2+self.mass**2)
            mag_field = magnetic_field(position)

            # Should accommodate for both the tests with constant stopping power and 
            # the actual simulation with beta dependent stopping power.

            stopping_power = get_stopping_power(density(position),stop_power,beta)

            # In order to keep track of the particle properties, we solve for 8 quantities. 
            # This WILL make bigger simulation slower.

            dpds = mom_ode(mag_field, stopping_power, direction, self.charge, beta)
            drds = pos_ode(direction)
            mag_work = mag_work_ode(mag_field,direction,self.charge)
            stop_work = stop_work_ode(stopping_power)
            return np.concatenate((drds, dpds, [mag_work, stop_work]))

        y0 = np.concatenate((self.position, self.momentum, [self.magnetic_work, self.stopping_work]))
        sol = solve_ivp(ode, [0, limit], y0, method='DOP853')

        # The idea is for the propagation to be done in small chunks, allowing us to check the postion of the particle 
        # (whether it has arrived on earth). The particle properties are updated after each stretch.

        self.position = sol.y[:3, -1]
        self.momentum = sol.y[3:6, -1]
        self.magnetic_work = sol.y[6,-1]
        self.stopping_work = sol.y[7, -1]

        p=np.linalg.norm(self.momentum)

        self.beta = p/np.sqrt(p**2+self.mass**2)
        self.lorentzFactor = 1/np.sqrt(1-self.beta**2)
        self.totalEnergy = self.lorentzFactor * self.mass
        self.kineticEnergy = self.totalEnergy - self.mass
        self.direction = self.momentum/np.linalg.norm(self.momentum)
