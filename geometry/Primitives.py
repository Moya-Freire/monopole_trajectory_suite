import numpy as np
import pandas as pd
from Material import Material 
from PrimitiveFunctions import *
from Compare import compare_varattr
from Serialization import *

# These are the primitives that will be used to create our earth model. The idea is that we 
# have shells, which can be used to represent both the Earth's strata and the atmosphere, 
# with support for changes in elevation, and spheres, which can be used to represent the Earth
# as solid. For an initial evaluation, since monopoles are not very penetrating, Earth can be 
# made using a sphere with variable radius given by DEMs and the atmosphere as a shell enveloping
# it, with variable density.

# The SphereShell class is the complicated son, and the one that actually 
# makes it possible to represent earth. It is capable of accepting variable
# thickness and radius at the same time, meaning that we can 

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

    # Equality support is given to each class so that we may track where a particle
    # is more easily. They have some limitations, such as considering that a constant
    # radius or density will always be given as a cosntant rather than the output of 
    # a function. Be aware of this 
    def __eq__(self,other):

        equal_material=self.material==other.material
        equal_position=np.all(self.position==other.position)
        equal_radius= compare_varattr(self.radius,other.radius)
        equal_thickness=compare_varattr(self.thickness,other.thickness)

        return equal_radius and equal_thickness and equal_position and equal_material

    def moveTo(self,position):
        self.position=position

    # This method makes it possible for us to change the material a given object is made 
    # of on the fly, alllowing us to test different materials for a same experimental
    # arragement

    def setMaterial(self,material):
        self.material=material

    # The user can also change both the radius and thickness of the spherical shell
    # without needing to declare a new shell

    def setRadius(self,radius):
        self.radius=radius

    def getRadius(self,theta,phi):
        if callable(self.radius):
            return self.radius(theta,phi)
        else:
            return self.radius
        
    def getRadiusType(self):
        if callable(self.radius):
            return "Variable"
        else:
            return "Constant"
        
    def setThickness(self,thickness):
        self.thickness=thickness
        
    def getThickness(self,theta,phi):
        if callable(self.thickness):
            return self.thickness(theta,phi)
        else:
            return self.thickness
        
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
    
    # These functions are the life and blood of the propagator, as the SimulationVolume
    # class uses them to check which density, and consequently stopping power, to return
    # to the propagator itself. As these are not really meant to be used by the user, 
    # but rather by the propagator, they assume the coordinate system and do the conversion
 
    def isInside(self,check):
        relative_position=check-self.position
        x,y,z=relative_position
        r,theta,phi=cartesian_to_spherical(x,y,z)
        return self.getRadius(theta,phi)>=r and (self.getRadius(theta,phi)-self.getThickness(theta,phi))<=r
    
    # These functions should provide the injector with a way to determine the vector
    # pointing from a given point in the surface of the object to its center, 
    # allowing us to randomize the monopole direction for injection, AKA the mess the user 
    # shouldn't have to deal with. 

    def getNormal(self,theta,phi,as_tuple=False):
        r=self.getRadius(theta,phi)-self.getThickness(theta,phi)
        x,y,z=spherical_to_cartesian(r,theta,phi,self.position)
        position=np.array([x,y,z])
        position_vector=position-self.position
        normal=position_vector/r
        if as_tuple:
            normal=(normal[0],normal[1],normal[2])
        return normal

    def getAntiNormal(self,theta,phi,as_tuple=False):
        return -1*self.getNormal(theta,phi,as_tuple)

    def getSurfacePoint(self,normal=False):
        theta=np.arccos(np.random.uniform(-1,1))
        phi=np.random.uniform(0,np.pi*2)
        r=self.getRadius(theta,phi)-self.getThickness(theta,phi)
        if normal:
            direction=self.getNormal(theta,phi)
        else:
            direction=self.getAntiNormal(theta,phi)
        x,y,z=spherical_to_cartesian(r,theta,phi)
        coords=np.array([x,y,z])
        return coords,direction
    
    def saveObject(self):

        
        thetas=np.arccos(np.linspace(-1,1,int(np.ceil(2/angular_resolution)+1)))
        phis=np.linspace(0,np.pi*2,int(np.ceil(2*np.pi/angular_resolution+1)))

        if callable(self.radius):
            serial_radius=grid_2D(self.getRadius,(thetas,phis),pack=False)
        else:
            serial_radius=self.radius

        if callable(self.thickness) or callable(self.radius):
            inner_radius=lambda theta,phi:self.getRadius(theta,phi)-self.getThickness(theta,phi)
            serial_inner_radius=grid_2D(inner_radius,(thetas,phis),pack=False)
        else:
            serial_inner_radius=self.radius-self.thickness

        if self.material.getDensityType()!="Variable":
            serial_density=self.material.density

        else:
            if not callable(self.radius):
                rs=np.linspace(0,self.radius,int(np.ceil(self.radius/radius_resolution)+1))
                serial_density=grid_3D(self.position,self.getDensity,(rs,thetas,phis),coord_system="Spherical")
            else:
                serial_density=var_grid3D(self.position,self.getDensity,serial_radius,
                                          radius_resolution,lower=serial_inner_radius,coord_system="Spherical")

        if self.material.preset is not None:
            name=self.material.preset
        else:
            name="Custom"

        Material_dict={"Material":name,
                       "Composition":self.material.composition,
                       "Density Type":self.material.getDensityType(),
                       "Density":serial_density
                       }

        return{"Solid":"SphereShell",
               "Radius Type":self.getRadiusType(),
               "Radius":serial_radius,
               "Inner Radius":serial_inner_radius,
               "Position":self.position,
               "Material":Material_dict}
    
class Sphere:
    def __init__(self,position,radius,material=None):
        self.radius=radius
        self.position=position
        self.material=material

    def __eq__(self, other):

        equal_radius=compare_varattr(self.radius,other.radius)
        equal_position=np.all(self.position==other.position)
        equal_material=self.material==other.material

        return equal_material and equal_position and equal_radius

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
        
    def getRadiusType(self):
        if callable(self.radius):
            return "Variable"
        else:
            return "Constant"
        
    def isInside(self,check):
        relative_position=check-self.position
        x,y,z=relative_position
        r,theta,phi=cartesian_to_spherical(x,y,z)
        return self.getRadius(theta,phi)>=r
    
    def getNormal(self,theta,phi,as_tuple=False):
        r=self.getRadius(theta,phi)
        x,y,z=spherical_to_cartesian(r,theta,phi,self.position)
        position=np.array([x,y,z])
        position_vector=position-self.position
        normal=position_vector/r
        if as_tuple:
            normal=(normal[0],normal[1],normal[2])
        return normal

    def getAntiNormal(self,theta,phi,as_tuple=False):
        return -1*self.getNormal(theta,phi,as_tuple)
    
    def getSurfacePoint(self,normal=False):
        theta=np.arccos(np.random.uniform(-1,1))
        phi=np.random.uniform(0,np.pi*2)
        r=self.getRadius(theta,phi)
        if normal:
            direction=self.getNormal(theta,phi)
        else:
            direction=self.getAntiNormal(theta,phi)
        x,y,z=spherical_to_cartesian(r,theta,phi)
        coords=np.array([x,y,z])
        return coords,direction

    def saveObject(self,radius_resolution,angular_resolution):
        angular_resolution=np.radians(angular_resolution/3600)

        thetas=np.arccos(np.linspace(-1,1,int(np.ceil(2/angular_resolution)+1)))
        phis=np.linspace(0,np.pi*2,int(np.ceil(2*np.pi/angular_resolution)+1))

        if callable(self.radius):
            serial_radius=grid_2D(self.getRadius,(thetas,phis),pack=False)
        else:
            serial_radius=self.radius

        if self.material.getDensityType()!="Variable":
            serial_density=self.material.density

        else:
            if not callable(self.radius):
                rs=np.linspace(0,self.radius,int(np.ceil(self.radius/radius_resolution)))
                serial_density=grid_3D(self.position,self.getDensity,(rs,thetas,phis),coord_system="Spherical")
            else:
                serial_density=var_grid3D(self.position,self.getDensity,serial_radius,
                                          radius_resolution,coord_system="Spherical")

        if self.material.preset is not None:
            name=self.material.preset
        else:
            name="Custom"

        Material_dict={"Material":name,
                       "Composition":self.material.composition,
                       "Density Type":self.material.getDensityType(),
                       "Density":serial_density
                       }

        return{"Solid":"Sphere",
               "Radius Type":self.getRadiusType(),
               "Radius":serial_radius,
               "Position":self.position,
               "Material":Material_dict}
    
# The block class is the ugly son, and is meant to provide us with a propagation space for
# simulation. As a result, it has no support for height maps.

class Block:
    def __init__(self,position,dimensions,material=None):
        self.dimensions=dimensions
        self.position=position
        self.material=material
        self.orientation={'x':0.0,
                          'y':0.0,
                          'z':0.0}
        self.vertexes=vertex(dimensions,position)
        self.rotation_history=[]
        self.faces=faces(self.vertexes)
        self.area=area(dimensions)

    def __eq__(self, other):

        equal_dimensions = np.all(self.dimensions==other.dimensions)
        equal_position = np.all(self.position==other.position)
        equal_material = self.material == other.material
        equal_orientation = self.orientation == other.orientation

        return equal_position and equal_material and equal_dimensions and equal_orientation

    def rotate(self,theta,axis):
        vertex_temp=self.vertexes-self.position
        self.rotation_history.append((axis,theta))
        rotation_matrix=rot_matrix(axis,theta)
        self.orientation[axis]+=theta
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

    # Determining whether a point is inside th block is less straightforward than
    # for the spherical objects, so the class keeps track of every rotation done to
    # the block and in order to check if a point is inside the block undoes all of 
    # them and compares the coordinates of the point to the dimensions of the block.

    def isInside(self,check):
        check=check-self.position
        check=revert_frame(check,self.rotation_history)
        check_x,check_y,check_z=check
        length,width,height=self.dimensions
        check_x=np.abs(check_x)>length/2
        check_y=np.abs(check_y)>width/2
        check_z=np.abs(check_z)>height/2

        return not (check_x or check_y or check_z)
    
    # As everything with the block class, getting the normal vector for the injector 
    # is not as straightforward as for the spherical objects, and we need to figure 
    # out on which face of the block a given point is, then calculate the vector based
    # on the vertexes for that face.

    def getFace(self,position):
        for face in self.faces:
            vec1=self.vertexes[face[0]]-self.vertexes[face[1]]
            vec2=self.vertexes[face[0]]-self.vertexes[face[2]]
            vec_test=self.vertexes[face[0]]-position
            if np.isclose(np.dot(vec1,np.cross(vec2,vec_test)),0):
                return [self.vertexes[face[0]],self.vertexes[face[1]],
                        self.vertexes[face[2]],self.vertexes[face[3]]]
    
    def getAntiNormal(self,position,as_tuple=False):
        face=self.getFace(position)
        vectors=[]

        for vertex in face:
            pos_vec=self.position-vertex
            vectors.append(pos_vec)

        dir=np.sum(np.array(vectors),axis=0)
        antinormal=dir/np.linalg.norm(dir)
        if as_tuple:
            antinormal=(antinormal[0],antinormal[1],antinormal[2])
        return antinormal

    def getNormal(self,position,as_tuple=False):
        return -1*self.getAntiNormal(position,as_tuple)
    
    def getSurfacePoint(self,normal=False):

        dimension,side=weighted_choice(self.dimensions)
        coords=np.zeros(3)
        coords[dimension]=self.dimensions[dimension]/2*(-1)**side

        for i in range(len(self.dimensions)):
            if i!=dimension:
                coords[i]=np.random.uniform(-self.dimensions[i]/2,self.dimensions[i]/2)

        coords=moveto_frame(coords,self.rotation_history)+self.position
        
        if normal:
            direction=self.getNormal(coords)
        else:
            direction=self.getAntiNormal(coords)

        coords=(coords[0],coords[1],coords[2])
        return coords, direction
            
    def saveObject(self,side_resolution):
        len_resolution,width_resolution,height_resolution=side_resolution
        l,w,h=self.dimensions

        lengths=np.linspace(-l/2,l/2,int(np.ceil(l/len_resolution)+1))
        widths=np.linspace(-w/2,w/2,int(np.ceil(w/width_resolution)+1))
        heights=np.linspace(-h/2,h/2,int(np.ceil(h/height_resolution)+1))

        if self.material.getDensityType()!="Variable":
            serial_density=self.material.density
        else:
            serial_density=grid_3D(self.position,self.getDensity,(lengths,widths,heights))

        if self.material.preset is not None:
            name=self.material.preset
        else:
            name="Custom"

        Material_dict={"Material":name,
                       "Composition":self.material.composition,
                       "Density Type":self.material.getDensityType(),
                       "Density":serial_density
                       }

        return{"Solid":"Block",
                "Dimensions":self.dimensions,
                "Position":self.position,
                "Orientation":self.orientation,
                "Rotation History":self.rotation_history,
               "Material":Material_dict}
    

        
