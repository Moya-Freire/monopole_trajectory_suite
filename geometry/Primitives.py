import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from Material import Material 
 
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

# These are the primitives that will be used to create our earth model. The idea is that we 
# have shells, which can be used to represent both the Earth's strata and the atmosphere, 
# with support for changes in elevation, and spheres, which can be used to represent the Earth
# as solid. For an initial evaluation, since monopoles are not very penetrating, Earth can be 
# made using a sphere with variable radius given by DEMs and the atmosphere as a shell enveloping
# it, with variable density.

class SphereShell:
    def __init__(self,position,radius,thickness,material=None):
        self.radius=radius
        self.thickness=thickness
        self.position=position

        # Every primitive must be declared with a material that composes it. 
        # The two classes are complementary, as the material class holds the density,
        # but doesn't know how to read it, and the primitive knows how to read the density,
        # but not what it is.

        self.material=material

    def moveTo(self,position):
        self.position=position

    def setMaterial(self,material):
        self.material=material

    # Make sure to correctly inform the function of which coordinate system your 
    # density function (and consequently your stopping power) uses, as failure to do 
    # so may result in runtime errors or the wrong output

    def getDensity(self,position,coord_system="Spherical"):
        relative_position=position-self.position
        x,y,z=relative_position
        if coord_system=="Spherical":
            r,theta,phi=cartesian_to_spherical(x,y,z)
            position=(r,theta,phi)
        else:
            position=(x,y,z)
        return self.material.getDensity(*position)
        
    def getStopPower(self,position,beta,charge,coord_system="Spherical"):
        relative_position=position-self.position
        x,y,z=relative_position
        if coord_system=="Spherical":
            r,theta,phi=cartesian_to_spherical(x,y,z)
            position=(r,theta,phi)
        else:
            position=(x,y,z)
        return self.material.getStopPower(position,beta,charge)
    
    def setRadius(self,radius):
        self.radius=radius

    def getRadius(self,theta=0,phi=0):
        if callable(self.radius):
            return self.radius(theta,phi)
        else:
            return self.radius

    # These functions are the life and blood of the propagator, as the SimulationVolume
    # class uses them to check which density, and consequently stopping power, to return
    # to the propagator itself. As these are not really meant to be used by the user, 
    # but rather by the propagator, they assume the coordinate system and do the conversion
 
    def isInside(self,check):
        relative_position=check-self.position
        x,y,z=relative_position
        r,theta,phi=cartesian_to_spherical(x,y,z)
        if callable(self.radius):
            return self.radius(theta,phi)>=r and (self.radius(theta,phi)-self.thickness)<=r
        else: 
            return self.radius>=np.linalg.norm(self.position-check) and (self.radius-self.thickness)<=r


class Sphere:
    def __init__(self,position,radius,material=None):
        self.radius=radius
        self.position=position
        self.material=material

    def moveTo(self,position):
        self.position=position

    def setMaterial(self,material):
        self.material=material

    def getDensity(self,position,coord_system="Spherical"):
        relative_position=position-self.position
        x,y,z=relative_position
        if coord_system=="Spherical":
            r,theta,phi=cartesian_to_spherical(x,y,z)
            position=(r,theta,phi)
        else:
            position=(x,y,z)
        return self.material.getDensity(*position)

    def getStopPower(self,position,beta,charge,coord_system="Spherical"):
        relative_position=position-self.position
        x,y,z=relative_position
        if coord_system=="Spherical":
            r,theta,phi=cartesian_to_spherical(x,y,z)
            position=(r,theta,phi)
        else:
            position=(x,y,z)
        return self.material.getStopPower(position,beta,charge)

    def setRadius(self,radius):
        self.radius=radius

    def getRadius(self,theta=0,phi=0):
        if callable(self.radius):
            return self.radius(theta,phi)
        else:
            return self.radius

    def isInside(self,check):
        if callable(self.radius):
            relative_position=check-self.position
            x,y,z=relative_position
            r,theta,phi=cartesian_to_spherical(x,y,z)
            return self.radius(theta,phi)>=r
        else: 
            return self.radius>=np.linalg.norm(self.position-check)

# The block class is the ugly son, and is meant to provide us with a propagation space for
# simulation. As a result, it has no support for height maps.

class Block:
    def __init__(self,position,dimensions,material=None):
        self.dimensions=dimensions
        self.position=position
        self.material=material
        self.orientation=np.array([0.0,0.0,0.0])
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

    def setMaterial(self,material):
        self.material=material

    def getDensity(self,position,coord_system="Cartesian"):
        relative_position=position-self.position
        x,y,z=relative_position
        if coord_system=="Spherical":
            r,theta,phi=cartesian_to_spherical(x,y,z)
            position=(r,theta,phi)
        else:
            position=(x,y,z)
        return self.material.getDensity(*position)
    
    def getStopPower(self,position,beta,charge,coord_system="Cartesian"):
        relative_position=position-self.position
        x,y,z=relative_position
        if coord_system=="Spherical":
            r,theta,phi=cartesian_to_spherical(x,y,z)
            position=(r,theta,phi)
        else:
            position=(x,y,z)
        return self.material.getStopPower(position,beta,charge)

    def setRadius(self,radius): 
        self.radius=radius

    def isInside(self,check):
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


