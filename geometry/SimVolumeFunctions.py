import numpy as np
import pandas as pd
from Primitives import Sphere,SphereShell,Block
from PrimitiveFunctions import spherical_to_cartesian,cartesian_to_spherical
import tqdm


# A lot of the functions defined here will look very convoluted at first. 
# The problem is that the process of creating an interpolation grid for the stopping power is
# extremely demanding, and for this reason we absolutely need to squeeze every bit of
# performance we can out of the code. For this reason, there are multiple nested loops 
# for exception handling. 

# I am making the assumption no one will give a constant density
# as a function, because if they do, they will lose performance.

def resolve_stoppower(simvolume,temp,position,medium,priority,beta,charge,cache_media):
    if medium.material.getDensityType()=="constant":
        key=(charge,beta)

        if key in cache_media[priority]:
            temp.append(cache_media[priority][(charge,beta)])
        else:
            cache=medium.getStopPower(position,beta,charge)
            cache_media[priority][(charge,beta)]=cache
            temp.append(cache)

    else:
        temp.append(simvolume.getStopPower(position,beta,charge,medium=medium))

def grid_populate(simvolume,grid,coords,betas,charges,cache_media,spherical_coords=False):
    
    coords1,coords2,coords3=coords

    for coord1 in tqdm.tqdm(coords1, desc="Generating SP Grid"):
        coords1_temp=[]

        for coord2 in coords2:
            coords2_temp=[]

            for coord3 in coords3:
                coords3_temp=[]

                if spherical_coords:
                    x,y,z=spherical_to_cartesian(coord1,coord2,coord3)
                    position=(x,y,z)
                else:
                    position=(coord1,coord2,coord3)

                medium=simvolume.getMedium(position)

                if medium.material.preset=="Vacuum":
                    coords3_temp=[np.zeros(len(betas)) for charge in charges]
                    coords2_temp.append(coords3_temp)
                    continue

                priority=simvolume.getPriority(medium)

                for charge in charges:
                    charge_temp=[]

                    for beta in betas:

                        resolve_stoppower(simvolume,charge_temp,position,medium,priority,beta,charge,cache_media)

                    coords3_temp.append(charge_temp)
                coords2_temp.append(coords3_temp)
            coords1_temp.append(coords2_temp)
        grid.append(coords1_temp)

#def save_grid(grid,coords1,coords2,coords3,betas,charges,filename):
#    with h5py.File(filename, 'W') as f:

def spherical_estimate_precision(function):
    value=[]

    thetas=np.arccos(np.radians(np.arange(0,180,1)))
    phis=np.radians(np.arange(0,360,1))

    for theta in thetas:
        for phi in phis:
            value.append(function(theta,phi))

    return np.mean(value)/100

def oom_estimate(object):  

    if isinstance(object,Block):
        len,width,height=object.dimensions
        resolutions=[len/100,width/100,height/100]

    elif isinstance(object,Sphere) or isinstance(object,SphereShell):
        resolutions=[] 

        if isinstance(object,SphereShell):  
            if not callable(object.thickness):
                resolutions.append(object.thickness/100)
            else:
                resolutions.append(spherical_estimate_precision(object.getThickness))

        else:
            if not callable(object.radius):
                resolutions.append(object.radius/100)
            else:
                resolutions.append(spherical_estimate_precision(object.getRadius))

        # Using a flat estimate of half a degree for the angular precision 
        resolutions.append(1800)

    return resolutions
