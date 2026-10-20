from Primitives import *
import numpy as np
from SimulationVolume import SimulationVolume
from EarthElevation import TerrainModel, spherical_to_latlong


# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
DEM_PATH = r"C:\Users\AYSAN\Desktop\monopole_trajectory_suite-main\ETOPO_2022_v1_30s_N90W180_surface.tif"   # <-- point this at your DEM (meters)
EXAGGERATION = 1           # plot-only vertical exaggeration
N_THETA, N_PHI = 360, 720   # plot resolution (1 degree by default)
SEA_LEVEL_FILL = 0.0        # height (m) used where the DEM has no data
VIEW = (25, -100)           # (elev, azim) of the camera

# ----------------------------------------------------------------------------
# WGS84 reference ellipsoid
# ----------------------------------------------------------------------------
A = 6_378_137.0             # equatorial radius, m
B = 6_356_752.314245        # polar radius, m
E2 = 1.0 - (B / A) ** 2     # first eccentricity squared

def ellipsoid_radius(theta):
    """Radius of the ellipsoid at geocentric colatitude theta (radians)."""
    theta = np.asarray(theta, dtype=float)
    return A * B / np.sqrt((B * np.sin(theta)) ** 2 + (A * np.cos(theta)) ** 2)

def geocentric_to_geodetic(lat_gc_deg):
    """Geocentric latitude (deg) -> geodetic latitude (deg)."""
    lat = np.radians(lat_gc_deg)
    return np.degrees(np.arctan2(np.sin(lat), (1.0 - E2) * np.cos(lat)))


def geodetic_to_spherical(lat_gd_deg, lon_deg):

    lat_gd = np.radians(lat_gd_deg)
    lat_gc = np.arctan2((1.0 - E2) * np.sin(lat_gd), np.cos(lat_gd))
    return np.pi / 2 - lat_gc, np.radians(lon_deg)

def elevation_lookup(terrain, lat, lon):
    x, y = terrain.to_raster.transform(lon, lat)
    dx = terrain.x[1] - terrain.x[0]
    dy = terrain.y[1] - terrain.y[0]
    xc = np.clip(x, terrain.x[0], terrain.x[-1])
    yc = np.clip(y, terrain.y[0], terrain.y[-1])
    x = np.where(np.abs(x - xc) <= dx, xc, x)
    y = np.where(np.abs(y - yc) <= dy, yc, y)

    return terrain.interpolator((y, x))

def make_earth(terrain, fill=SEA_LEVEL_FILL):
    def radius(theta, phi):
        theta, phi = np.broadcast_arrays(
            np.asarray(theta, dtype=float), np.asarray(phi, dtype=float)
        )
        lat_gc, lon = spherical_to_latlong(theta, phi)
        lat_gd = geocentric_to_geodetic(lat_gc)

        h = np.reshape(elevation_lookup(terrain, lat_gd, lon), theta.shape)
        h = np.where(np.isfinite(h), h, fill)  # NaN / out-of-bounds -> fill
        return ellipsoid_radius(theta) + h

    return Sphere(position=np.zeros(3), radius=radius)
