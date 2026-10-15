from Monopole import Monopole
from MonopoleAnimation import *
import numpy as np
import matplotlib.pyplot as plt

beta = 1e-3
charge = 1
mass = 1e7

field = lambda r: 1e-5*np.array([1,0,0])
stopping_power= 0
density=lambda r: 1

direction1 = np.array([1,0,0])
direction2 = np.array([1/np.sqrt(2),1/np.sqrt(2),0])
direction3 = np.array([0,1,0])
position = np.array([0,0,0])

monopole1 = Monopole( beta, position, direction1, mass,charge)
monopole2 = Monopole( beta, position, direction2, mass,charge)
monopole3 = Monopole( beta, position, direction3, mass,charge)

r01 = monopole1.position
r02 = monopole2.position
r03 = monopole3.position
p01 = monopole1.direction
po2 = monopole2.direction
po3 = monopole3.direction

position_x1=[r01[0]]
position_y1=[r01[1]]
position_z1=[r01[2]]
kinectic_energy1=[monopole1.kineticEnergy]
direction1=[p01]

position_x2=[r02[0]]
position_y2=[r02[1]]
position_z2=[r02[2]]
kinectic_energy2=[monopole2.kineticEnergy]
direction2=[po2]

position_x3=[r03[0]]
position_y3=[r03[1]]
position_z3=[r03[2]]
kinectic_energy3=[monopole3.kineticEnergy]
direction3=[po3]

traversed=0
while traversed<1e3:
    monopole1.propagate(field,stopping_power,density,50)
    monopole2.propagate(field,stopping_power,density,50)
    monopole3.propagate(field,stopping_power,density,50)

    pos_temp1=monopole1.position
    position_x1.append(pos_temp1[0])
    position_y1.append(pos_temp1[1])
    position_z1.append(pos_temp1[2])

    pos_temp2=monopole2.position
    position_x2.append(pos_temp2[0])
    position_y2.append(pos_temp2[1])
    position_z2.append(pos_temp2[2])

    pos_temp3=monopole3.position
    position_x3.append(pos_temp3[0])
    position_y3.append(pos_temp3[1])
    position_z3.append(pos_temp3[2])


    kinectic_energy1.append(monopole1.kineticEnergy)
    direction1.append(monopole1.direction)

    kinectic_energy2.append(monopole2.kineticEnergy)
    direction2.append(monopole2.direction)

    kinectic_energy3.append(monopole3.kineticEnergy)
    direction3.append(monopole3.direction)  

    traversed+=50

plt.plot(position_x1,position_y1,label=f'Parallel to field')
plt.plot(position_x2,position_y2,label=f'45 degree to field')
plt.plot(position_x3,position_y3,label=f'Perpendicular to field')
plt.xlabel('log x [log_10(m)]')
plt.ylabel('log y [log_10(m)]')
#plt.xscale('log')
#plt.yscale('log')
plt.grid()
plt.title('Monopole propagation test : Uniform magnetic field')
plt.legend()
plt.show()

positions = [
    np.column_stack((position_x1, position_y1, position_z1)),
    np.column_stack((position_x2, position_y2, position_z2)),
    np.column_stack((position_x3, position_y3, position_z3))
]

directions = [
    np.array(direction1),
    np.array(direction2),
    np.array(direction3)
]

energies = [
    np.array(kinectic_energy1),
    np.array(kinectic_energy2),
    np.array(kinectic_energy3)
]

animate_monopoles(
    positions=positions,
    directions=directions,
    energies=energies,
    field=1e-5 * np.array([1, 0, 0]),
    filename="monopole_propagation.gif",
    interval=100,
    arrow_length=40
)
