
import numpy as np
import rasterio
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator

EarthRadius= 6_371_200.0

# Chat GPT's code for extracting variations in the Earth's elevation at different locations.
# Should be able to generate, with the correct unit conversion, a height map to be used with
# the Sphere class defined in primitives, allowing us to represent Earth as basically a 
# bumpy ball 

def latlong_to_spherical(lat,long):
    lat,long=np.radians([lat,long])
    return (np.pi/2 - lat),long

def spherical_to_latlong(theta,phi):
    lat=np.degrees(np.pi/2-theta)
    long=np.degrees(phi)
    long=(long + 180.0) % 360.0 - 180.0
    return lat,long

class TerrainModel:
    def __init__(self, filepath):
        with rasterio.open(filepath) as dem:
            self.elevation = dem.read(
                1, masked=True
            ).astype(float).filled(np.nan)

            self.transform = dem.transform
            self.crs = dem.crs
            self.bounds = dem.bounds

            rows = np.arange(dem.height)
            cols = np.arange(dem.width)

            # Pixel-centre coordinates in the raster CRS.
            self.x = (
                self.transform.c
                + (cols + 0.5) * self.transform.a
            )
            self.y = (
                self.transform.f
                + (rows + 0.5) * self.transform.e
            )

        # Interpolator expects ascending coordinate arrays.
        if self.y[0] > self.y[-1]:
            self.y = self.y[::-1]
            self.elevation = self.elevation[::-1, :]

        if self.x[0] > self.x[-1]:
            self.x = self.x[::-1]
            self.elevation = self.elevation[:, ::-1]

        self.interpolator = RegularGridInterpolator(
            (self.y, self.x),
            self.elevation,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )

        self.to_raster = Transformer.from_crs(
            "EPSG:4326",
            self.crs,
            always_xy=True,
        )

    def getElevation(self, latitude, longitude):
        x, y = self.to_raster.transform(
            longitude, latitude
        )

        return float(self.interpolator((y, x)))
