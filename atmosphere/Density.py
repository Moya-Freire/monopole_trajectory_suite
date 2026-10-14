import numpy as np
import pandas as pd
import pymsis
SITES = {
"concordia": (-75.10, 123.35, 3.23),
"alert": ( 82.50, -62.35, 0.03),
"lhaaso": ( 29.36, 100.14, 4.41),
}
dates = np.array([np.datetime64("2025-01-15T00:00")])
altitudes_km = np.arange(0.0, 150.25, 0.25)
for name, (latitude, longitude, detector_altitude) in SITES.items():
    data = pymsis.calculate(dates, longitude, latitude, altitudes_km)
    density = data[0, 0, 0, :, 0] # total mass density, kg/m^3
    pd.DataFrame({
    "altitude_m": 1000.0 * altitudes_km,
    "density_kg_m3": density,
    }).to_csv(f"atmosphere_{name}.csv", index=False)
