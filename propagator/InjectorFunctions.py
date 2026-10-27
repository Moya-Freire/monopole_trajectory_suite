import numpy as np 
from Monopole import *
from PrimitiveFunctions import *


def propagation_vars(monopole,betas,positions, directions,volume):
        if isinstance(betas,list):
            beta=betas[monopole]
        else:
            beta=betas

        if positions is None or directions is None:
            if positions is None:
                position,normal=volume.getSurfacePoint()
                if directions is not None:
                    direction=directions[monopole]
                else:
                    direction=random_direction(normal)

            elif positions is not None and directions is None:
                position=positions[monopole]
                normal=volume.getAntiNormal(position)

                if normal is None:
                    raise ValueError("Monopole starting positions must" \
                    " be at the boundary of the simulation volume for correct simulation")
                else:
                    direction=random_direction(normal)
        else:
            position=positions[monopole]
            direction=directions[monopole]
        return beta,position,direction

def check_valid(betas,monopoles):

    if not isinstance(monopoles,list) and not isinstance(monopoles,int) and monopoles is not None:
        raise ValueError("Please provide either the number of monopoles " \
            "to be generated or a list with the pre-defined monopoles.")
    
    if not isinstance(betas,list) and not isinstance(betas,float):
        raise ValueError("Please provide the speed for the monopoles as either " \
        "an uniform speed or an array of beta values")

def random_direction(normal):
    x,y,z=normal[0],normal[1],normal[2]
    rn,tn,pn=cartesian_to_spherical(x,y,z)

    theta= np.arccos(np.random.uniform(0,1))
    phi = np.random.uniform(0,2*np.pi)
    x,y,z=spherical_to_cartesian(1,theta,phi)
    vec=np.array([x,y,z])

    rot_y=rot_matrix('y',tn)
    vec=vec@rot_y.T
    
    rot_z=rot_matrix('z',pn)
    vec=vec@rot_z.T

    return vec

def propagate_one(monopole,field,simvolume,chunk):
    while monopole.beta>1e-6 and simvolume.isContained(monopole.position):
                monopole.propagate(
                    field,
                    simvolume.getStopPower,
                    simvolume.getDensity,
                    limit=chunk
                )

def generate_monopoles(SimSpace,monopoles,betas,mass,charge,positions,directions):
    volume=SimSpace.medium
    particles=[]

    for monopole in range(monopoles):\
    
        beta,position,direction=propagation_vars(monopole,betas,positions,directions, volume)
        particles.append(Monopole(beta,position,direction,mass,charge))

    return particles
