import numpy as np 
from Monopole import *
from PrimitiveFunctions import *
from datetime import datetime

def propagation_vars(monopole,masses,charges,betas,positions, directions,volume):
        
        if isinstance(betas,list) or isinstance(betas,np.ndarray):
            beta=betas[monopole]
        else:
            beta=betas

        if isinstance(charges,list) or isinstance(charges,np.ndarray):
            charge = charges[monopole]
        else:
            charge = charges

        if isinstance(masses,list) or isinstance(masses,np.ndarray):
            mass = masses[monopole]
        else:
            mass = masses


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
        return mass,charge,beta,position,direction

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

def propagate_one(monopole,id,field,simvolume,chunk,save_history,condition):

    mass=monopole.mass
    charge=monopole.charge

    p0=monopole.position
    v0=monopole.beta
    E0=monopole.kineticEnergy
    dir0=monopole.direction

    position_history=[p0]
    beta_history=[v0]
    energy_history=[E0]
    direction_history=[dir0]
    magwork_history=[0]
    stopwork_history=[0]

    while (monopole.beta>1e-6 and simvolume.isContained(monopole.position)) or (condition and simvolume.isContained(monopole.position)):
                
                monopole.propagate(
                    field,
                    simvolume.getStopPower,
                    limit=chunk
                )
                if save_history:
                    position_history.append(monopole.position)
                    beta_history.append(monopole.beta)
                    energy_history.append(monopole.kineticEnergy)
                    direction_history.append(monopole.direction)
                    magwork_history.append(monopole.magnetic_work)
                    stopwork_history.append(monopole.stopping_work)

    pf=monopole.position
    vf=monopole.beta
    Ef=monopole.kineticEnergy
    dirf=monopole.direction

    mag_work=monopole.magnetic_work
    stop_work=monopole.stopping_work

    contained=simvolume.isContained(pf)

    p=(p0,pf,position_history)
    v=(v0,vf,beta_history)
    E=(E0,Ef,energy_history)
    dir=(dir0,dirf,direction_history)

    return monopole,monopole_dataset(id,
                            mass,charge,
                            p,
                            E,
                            v,
                            dir,
                            mag_work,magwork_history,
                            stop_work,stopwork_history,
                            contained,
                            save_history)

def generate_monopoles(SimSpace,monopoles,betas,masses,charges,positions,directions):

    volume=SimSpace.medium
    particles=[]

    for monopole in range(monopoles):
    
        mass,charge,beta,position,direction=propagation_vars(monopole,masses,charges,betas,
                                                             positions,directions, volume)
        particles.append(Monopole(beta,position,direction,mass,charge))

    return particles

def monopole_dataset(id,mass,charge,p,E,v,dir,mw,mw_history,sp,sp_history,contained,save_history):

    p0,pf,pos_history=p
    E0,Ef,en_history=E
    v0,vf,v_history=v
    dir0,dirf,dir_history=dir

    dataset={"ID":id,
             "Mass":mass,
             "Charge":charge,
            "Initial position":p0,
            "Final position":pf,
            "Initial energy":E0,
            "Final energy":Ef,
            "Initial velocity":v0,
            "Final velocity":vf,
            "Initial direction":dir0,
            "Final direction":dirf,
            "Magnetic work": mw,
            "Stopping power work":sp,
            "Contained":contained,
        }
    
    if save_history:

        dataset["Position history"] = pos_history
        dataset["Direction history"] = dir_history

        dataset["Energy history"] = en_history
        dataset["Velocity history"] = v_history

        dataset["Magnetic work history"]=mw_history
        dataset["Stopping power work history"]=sp_history

    return dataset

def metadata_datetime():
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
