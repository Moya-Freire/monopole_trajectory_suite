import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

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

# The idea is: You create a monopole object, then this same monopole
# can be propagated for different conditions, without the need for new code.
# This should allow us to study the effects of the magnetic field and 
# the atmosphere quite easily.

class Monopole:
    def __init__(self, beta, position, direction, mass,charge, mag_work=0, stop_work=0):
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

    def propagate(self,magnetic_field, stop_power,limit):
        
        def ode(s, y):
            position = y[:3]
            direction = y[3:6] /np.linalg.norm(y[3:6])
            p= np.linalg.norm(y[3:6])
            beta= p/np.sqrt(p**2+self.mass**2)
            mag_field = magnetic_field(position)
            stopping_power = stop_power(position)
            dpds = mom_ode(mag_field, stopping_power, direction, self.charge, beta)
            drds = pos_ode(direction)
            mag_work = mag_work_ode(mag_field,direction,self.charge)
            stop_work = stop_work_ode(stopping_power)
            return np.concatenate((drds, dpds, [mag_work, stop_work]))

        y0 = np.concatenate((self.position, self.momentum, [self.magnetic_work, self.stopping_work]))
        sol = solve_ivp(ode, [0, limit], y0, method='DOP853')

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
