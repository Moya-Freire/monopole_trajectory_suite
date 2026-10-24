import numpy as np
import pandas as pd
from Primitives import Sphere,SphereShell,Block

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

    def getStopPower(self,position,beta,charge):
        medium=self.getMedium(position)
        return medium.getStopPower(position,beta,charge)