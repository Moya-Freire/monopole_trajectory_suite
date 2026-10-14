import numpy as np
import pandas as pd

class TabulatedAtmosphere:
    def __init__(self, filename, earth_radius_m=6_371_200.0):
        data = np.loadtxt(filename, delimiter=",", skiprows=1)
        self.altitude_m = data[:, 0]
        self.density = data[:, 1]
        self.earth_radius_m = earth_radius_m
    def density_kg_m3(self, position_m, time_s=0.0):
        altitude = np.linalg.norm(position_m) - self.earth_radius_m
        if altitude > self.altitude_m[-1]:
            return 0.0
        altitude = max(altitude, self.altitude_m[0])
        return np.interp(altitude, self.altitude_m, self.density)
    def composition(self, position_m, time_s=0.0):
        return "dry_air"
