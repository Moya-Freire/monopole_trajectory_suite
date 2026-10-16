import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

def vertex(dimensions,position):
    vertexes=[]
    length,width,height=dimensions
    vertexes.append(position+np.array([length/2,width/2,height/2]))
    vertexes.append(position+np.array([-length/2,width/2,height/2]))
    vertexes.append(position+np.array([-length/2,-width/2,height/2]))
    vertexes.append(position+np.array([length/2,-width/2,height/2]))
    vertexes.append(position+np.array([length/2,width/2,-height/2]))
    vertexes.append(position+np.array([-length/2,width/2,-height/2]))
    vertexes.append(position+np.array([-length/2,-width/2,-height/2]))
    vertexes.append(position+np.array([length/2,-width/2,-height/2]))
    return vertexes

def cartesian_to_spherical(x,y,z):
    r=np.linalg.norm([x,y,z])
    theta=np.arccos(z/r) if r>0 else 0
    phi=np.arctan2(y,x)
    return r,theta,phi


class SphereShell:
    def __init__(self,position,radius,thickness,density=0,stop_power=0):
        self.radius=radius
        self.thickness=thickness
        self.position=position
        self.density=density
        self.stop_power=stop_power

    def moveTo(self,position):
        self.position=position

    def setMaterial(self,density,stop_power):
        self.density=density
        self.stop_power=stop_power

    def getDensity(self,position):
        if callable(self.density):
            relative_position=position-self.position
            x,y,z=relative_position
            r,theta,phi=cartesian_to_spherical(x,y,z)
            return self.density(r,theta,phi)
        else:
            return self.density

    def getStopPower(self,beta=0):
        if callable(self.stop_power):
            return self.stop_power(beta)
        else:
            return self.stop_power
        
    def setRadius(self,radius):
        self.radius=radius

    def getRadius(self,theta=0,phi=0):
        if callable(self.radius):
            return self.radius(theta,phi)
        else:
            return self.radius

    def checkInside(self,check):
        if callable(self.radius):
            relative_position=check-self.position
            x,y,z=relative_position
            r,theta,phi=cartesian_to_spherical(x,y,z)
            return self.radius(theta,phi)>r and (self.radius(theta,phi)-self.thickness)<r
        else: 
            return self.radius>np.linalg.norm(self.position-check) and (self.radius-self.thickness)<r


class Sphere:
    def __init__(self,position,radius,density=0,stop_power=0):
        self.radius=radius
        self.position=position
        self.density=density
        self.stop_power=stop_power

    def moveTo(self,position):
        self.position=position

    def setMaterial(self,density,stop_power):
        self.density=density
        self.stop_power=stop_power

    def getDensity(self,position):
        if callable(self.density):
            relative_position=position-self.position
            x,y,z=relative_position
            r,theta,phi=cartesian_to_spherical(x,y,z)
            return self.density(r,theta,phi)
        else:
            return self.density

    def getStopPower(self,beta=0):
        if callable(self.stop_power):
            return self.stop_power(beta)
        else:
            return self.stop_power

    def setRadius(self,radius):
        self.radius=radius

    def getRadius(self,theta=0,phi=0):
        if callable(self.radius):
            return self.radius(theta,phi)
        else:
            return self.radius

    def checkInside(self,check):
        if callable(self.radius):
            relative_position=check-self.position
            x,y,z=relative_position
            r,theta,phi=cartesian_to_spherical(x,y,z)
            return self.radius(theta,phi)>r
        else: 
            return self.radius>np.linalg.norm(self.position-check)

class Block:
    def __init__(self,position,dimensions,density=0,stop_power=0):
        self.dimensions=dimensions
        self.position=position
        self.density=density
        self.orientation=np.array([0,0,0])
        self.stop_power=stop_power
        self.vertexes=vertex(dimensions,position)
        self.rotation_history=[]

    def rotate(self,theta,axis):
        vertex_temp=self.vertexes-self.position

        if axis=='x':
            self.rotation_history.append(('x',theta))
            self.orientation[0]+=theta
            rotation_matrix=np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])
        elif axis=='y':
            self.rotation_history.append(('y',theta))
            self.orientation[1]+=theta
            rotation_matrix=np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
        elif axis=='z':
            self.rotation_history.append(('z',theta))
            self.orientation[2]+=theta
            rotation_matrix=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
        
        self.vertexes=vertex_temp@rotation_matrix.T+self.position

    def moveTo(self,position):
        self.position=position
        self.vertexes=vertex(self.dimensions,position)

    def setMaterial(self,density,stop_power):
        self.density=density
        self.stop_power=stop_power

    def setRadius(self,radius): 
        self.radius=radius

    def checkInside(self,check):
        check=check-self.position
        vertex_temp=self.vertexes-self.position

        for rotation in reversed(self.rotation_history):
            axis,theta=rotation
            theta=-theta
            if axis=='x':
                rotation_matrix=np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])
            elif axis=='y':
                rotation_matrix=np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
            elif axis=='z':
                rotation_matrix=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
            
            vertex_temp=vertex_temp@rotation_matrix.T
            check=check@rotation_matrix.T

        check_x,check_y,check_z=check
        length,width,height=self.dimensions
        check_x=np.abs(check_x)>length/2
        check_y=np.abs(check_y)>width/2
        check_z=np.abs(check_z)>height/2
        return not (check_x or check_y or check_z)


