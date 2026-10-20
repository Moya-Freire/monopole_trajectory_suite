from Primitives import *
import numpy as np
from SimulationVolume import SimulationVolume
from EarthElevation import TerrainModel, spherical_to_latlong
from EarthMaker import make_earth
from AtmosphereDensity import AtmosphereDensity
DEM_PATH = r"C:\Users\AYSAN\Desktop\monopole_trajectory_suite-main\ETOPO_2022_v1_30s_N90W180_surface.tif"   # <-- point this at your DEM (meters)
DENSITY_PATH = r"C:\Users\AYSAN\Desktop\monopole_trajectory_suite-main\atmosphere.h5"
EarthRadius= 6_371_200.0
def EarthModel(DEM_PATH,DENSITY_PATH):
    density=AtmosphereDensity(DENSITY_PATH)
    terrain = TerrainModel(DEM_PATH)
    Earth=make_earth(terrain)
    Atmosphere=SphereShell(np.zeros(3),EarthRadius+150*1000,150*1000,density,stop_power=0.1)
    PropagationVolume=SimulationVolume(0,10*EarthRadius,shape='Sphere',object_list=[Earth,Atmosphere])
    return PropagationVolume
