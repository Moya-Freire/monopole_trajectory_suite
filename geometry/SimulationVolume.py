import numpy as np
import pandas as pd
from Primitives import Sphere,SphereShell,Block
from PrimitiveFunctions import spherical_to_cartesian,cartesian_to_spherical
import h5py
import tqdm
from SimVolumeFunctions import *
import warnings
from Serialization import spherical_estimate_precision

default_resolution={ "Sphere": [1e3, 18, 36],
                    "Block": [1e2,1e2,1e2]
                    }

class SimulationVolume:
    def __init__(self,material,dimensions,shape,object_list=None):

        self.object_list=object_list if object_list is not None else []

        if shape=='Sphere':
            self.medium=Sphere(np.array([0,0,0]),dimensions,material)
        elif shape=='Block':
            self.medium=Block(np.array([0,0,0]),dimensions,material)
        else:
            raise ValueError("Propagation space must be defined as " \
            "either a cuboid or a spherical solid")

    def addObject(self,object,priority):
        self.object_list.append((priority,object))

    def isContained(self,position): 
        return self.medium.isInside(position)

    def getPriority(self,medium):
        # Per definition, the "background" medium, given by the simulation volume itself,
        # has the lowest priority.

        if medium==self.medium:
            return 0
        
        for priority,object in self.object_list:
            if object==medium:
                return priority

    def getMedium(self,position):

        medium=None
        medium_priority=-np.inf

        for priority,object in self.object_list:
            if object.isInside(position):
                if priority>medium_priority:
                    medium_priority=priority
                    medium=object

        if medium is not None:
            return medium
        else:
            return self.medium

    def getDensity(self,position):
        medium=self.getMedium(position)
        return medium.getDensity(position)

    def getStopPower(self,position,beta,charge,medium=None):
        if medium is None:
            medium=self.getMedium(position)
        return medium.getStopPower(position,beta,charge)

    def getVelocityLimit(self,position):
        medium=self.getMedium(position)
        if medium.material.preset=="Vacuum":
            return -np.inf
        else:
            return 1e-6

    def saveSimVolume(self,resolutions=None):


    # This function is used to generate a grid of stopping power values for 
    # a given simulation volume to be used for interpolation. This is a COSTLY
    # process, even though is meant to make the actual execution of the simulation faster.
    # It does not make much sense to do it for a simulation volume that won't 
    # be used multiple times or for small batches of monopoles. It is best to only call it
    # when using a cluster.

    # Default values are very conservative, meaning they will generate a very granular grid 
    # at the expense of a very big resulting HDF5 file. Unless you have violent changes in 
    # density with the position, might not be needed. This is a function that will be useful,
    # but requires some thought to be used properly. 

    def getSPGrid(self,filename="SPGrid.h5",beta_resolution=1e3,
                  dimension_resolution=None,max_abscharge=3):

        if dimension_resolution is None:
            resolution=default_resolution[self.medium.shape]
        else:
            resolution=dimension_resolution

        betas=np.logspace(-6,0,beta_resolution)
        charges=np.arange(1,max_abscharge+1,1)
        SP_grid=[]
        cache_media=[{} for priority,object in self.object_list]

        print("Generating stopping power grid for simulation volume. Sit tight, " \
        "this will take a while...")

        if self.medium.shape=="Sphere":
            
            rs=np.linspace(0,self.dimensions,int(resolution[0]))
            thetas=np.arccos(np.linspace(-1,1,int(resolution[1])))
            phis=np.linspace(0,2*np.pi,int(resolution[2]))

            grid_populate(self,SP_grid,(rs,thetas,phis),betas,charges,cache_media,spherical_coords=True)

        if self.medium.shape=="Block":
            xs=np.linspace(-self.dimensions[0]/2,self.dimensions[0]/2,int(resolution[0]))
            ys=np.linspace(-self.dimensions[1]/2,self.dimensions[1]/2,int(resolution[1]))
            zs=np.linspace(-self.dimensions[2]/2,self.dimensions[2]/2,int(resolution[2]))

            grid_populate(self,SP_grid,(xs,ys,zs),betas,charges,cache_media,spherical_coords=False)