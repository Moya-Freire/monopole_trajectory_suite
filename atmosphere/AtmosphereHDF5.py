import numpy as np
import pandas as pd
import pymsis
import h5py

lat=np.linspace(-90,90,361)
long=np.linspace(-180,180,720)
coords=zip(lat,long)
dates = np.array([np.datetime64("2025-01-15T00:00")])
altitudes_km = np.arange(0.0, 150.25, 0.25)
densities={"Altitude":1000*altitudes_km}

with h5py.File("atmosphere.h5",'w') as f:
  h5_latitude=f.create_dataset("latitude",data=lat)
  h5_latitude.attrs['units']='degrees'

  h5_longitude=f.create_dataset("longitude",data=long)
  h5_longitude.attrs['units']='degrees'

  h5_altitude=f.create_dataset("altitude",data=(1000*altitudes_km))
  h5_altitude.attrs['units']='meters'

  h5_density = f.create_dataset(
    "density",
    shape=(len(lat), len(long), len(altitudes_km)),
    dtype="float64",
    chunks=(1, 1, len(altitudes_km)),
    compression="lzf",
    shuffle=True,
  ) 
  h5_density.attrs['units']='kg/m^3'

  for i,latitude in enumerate(lat):
    for j,longitude in enumerate(long):
      data = pymsis.calculate(dates, longitude, latitude, altitudes_km)
      h5_density[i,j,:] = data[0, 0, 0, :, 0]
    print(f"Completed latitude: {i}/{len(lat)}")
