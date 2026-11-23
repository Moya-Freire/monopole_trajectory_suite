import numpy as np
import pandas as pd

# Coordinate conversion stuff

def cartesian_to_spherical(x,y,z):
    r=np.linalg.norm([x,y,z],axis=1)
    theta=np.arccos(np.divide(z,r,out=np.zeros_like(z,dtype=float),where=r!=0))
    phi=np.arctan2(y,x)%(2*np.pi)
    return r,theta,phi

def spherical_to_cartesian(r,theta,phi,position=np.array([0,0,0])):

    x=r*np.sin(theta)*np.cos(phi)+position[0]
    y=r*np.sin(theta)*np.sin(phi)+position[1]
    z=r*np.cos(theta)+position[2]

    return x,y,z

 # Block class stuff

 # This very weird function is responsible for defining the vertexes
# of the block block in space, which will act as anchors for both 
# rotating and determining whether a point is inside the block

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

def faces(vertexes):
    faces=[]
    anchor={0:[],
            1:[],
            2:[],}
    for i in range(len(vertexes)):

        for j in range(len(vertexes[i])):

            if vertexes[i][j] not in anchor[j]:
                face=[i]

                for k in range(len(vertexes)):
                    if not np.all(vertexes[i]==vertexes[k]):
                        if vertexes[k][j]==vertexes[i][j]:
                            face.append(k)

                anchor[j].append(vertexes[i][j])
                face=np.array(face)
                faces.append(face)
    return faces

def rot_matrix(axis,theta):
    if axis=='x':
        rotation_matrix=np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])
    
    elif axis=='y':
        rotation_matrix=np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
    
    elif axis=='z':
        rotation_matrix=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
    
    return rotation_matrix

def revert_frame(check, rotation_history):
    for rotation in reversed(rotation_history):
        axis,theta=rotation
        theta=-theta
        rotation_matrix=rot_matrix(axis,theta)
        check=check@rotation_matrix.T
    return check

def moveto_frame(check, rotation_history):
    for rotation in rotation_history:
        axis,theta=rotation
        rotation_matrix=rot_matrix(axis,theta)
        check=check@rotation_matrix.T
    return check

def area(dimensions):
    return dimensions[0]*dimensions[1]*2+dimensions[0]*dimensions[2]*2+dimensions[1]*dimensions[2]*2

def weighted_choice(dimensions):
    face_area=np.array([dimensions[1]*dimensions[2]*2,
                        dimensions[0]*dimensions[2]*2,
                        dimensions[0]*dimensions[1]*2])
    
    chance=0
    frac_area=face_area/area(dimensions)
    choice=[0,frac_area[0],np.sum(frac_area[:2]),np.sum(frac_area[:3])]
        
    chance=0
    while chance==0:
        chance=np.random.uniform()
    side=np.random.choice(2)

    for i in range(1,4):
        if choice[i]>=chance and choice[i-1]<chance:
            dimension=i-1      

    return dimension,side

