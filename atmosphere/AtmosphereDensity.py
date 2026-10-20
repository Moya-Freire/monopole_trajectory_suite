import numpy as np
import pandas as pd
from Primitives import Sphere,SphereShell,Block
import h5py
from scipy.interpolate import RegularGridInterpolator

EarthRadius= 6_371_200.0

class AtmosphereDensity:
    def __init__(self,filename):
        with h5py.File(filename,'r') as f:
            self.latitude=f['latitude'][:]
            self.longitude=f['longitude'][:]
            self.altitude=f['altitude'][:]
            self.density=f['density'][:]

            self.latitude=np.append(self.latitude,180)
            self.density=np.concatenate((self.density,self.density[:,:1,:]),axis=1)

            self.interpolator=RegularGridInterpolator(
                (self.latitude,self.longitude,self.altitude),
                self.density,
                method='linear',
                bounds_error=False,
                fill_value=np.nan,

            )
    
    def getDensity(self,r,theta,phi):
        altitude=r-EarthRadius
        longitude=np.degrees(phi)
        latitude=90-np.degrees(theta)
        longitude=(longitude + 180.0) % 360.0 - 180.0
        return(self.interpolator(latitude,longitude,altitude))

