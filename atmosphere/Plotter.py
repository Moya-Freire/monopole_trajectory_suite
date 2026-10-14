from AtmosphereInterpolator import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid as integrate
from scipy.integrate import quad

earth_radius_m=6_371_200.0

height=np.linspace(0,150000,1000)
mask_concordia=height>=3.23*1000
mask_lha=height>=4.41*1000
mask_alert=height>=30
height=height+earth_radius_m

atm_concordia=TabulatedAtmosphere("atmosphere_concordia.csv")
atm_lha=TabulatedAtmosphere("atmosphere_lhaaso.csv")
atm_alert=TabulatedAtmosphere("atmosphere_alert.csv")

exponential= lambda h:1.225*np.exp(-h/8500)
density_concordia=[]
density_lha=[]
density_alert=[]

for altitude in height:
    if altitude>earth_radius_m+3.23*1000:
        density_concordia.append(atm_concordia.density_kg_m3(altitude))
    if altitude>earth_radius_m+4.41*1000:
        density_lha.append(atm_lha.density_kg_m3(altitude))
    if altitude>earth_radius_m+30:
        density_alert.append(atm_alert.density_kg_m3(altitude))

print(density_concordia)
altitude_plot=height/1000-earth_radius_m/1000

integral_concordia = integrate(
    [atm_concordia.density_kg_m3(h) for h in height[mask_concordia]],
    x=height[mask_concordia]
) / 10

integral_lha = integrate(
    [atm_lha.density_kg_m3(h) for h in height[mask_lha]],
    x=height[mask_lha]
) / 10

integral_alert = integrate(
    [atm_alert.density_kg_m3(h) for h in height[mask_alert]],
    x=height[mask_alert]
) / 10
integral_exponential=quad(lambda h: exponential(h),0,150000)[0]/10

plt.plot(altitude_plot[mask_concordia],density_concordia,label=f"Concordia:{integral_concordia:.2f} g/cm^2")
plt.plot(altitude_plot[mask_lha],density_lha,label=f"LHAASO:{integral_lha:.2f} g/cm^2")
plt.plot(altitude_plot[mask_alert],density_alert,label=f"Alert:{integral_alert:.2f} g/cm^2")
plt.plot(altitude_plot,exponential(height),label=f"Sea level:{integral_exponential:.2f} g/cm^2 ")
plt.grid()
plt.title("Atmospheric density vs altitude")
plt.yscale("log")
plt.xlabel("log Altitude [log_10(km)]")
plt.ylabel("Density [kg/m^3]")
plt.legend()
plt.savefig("atmosphere_density.png",dpi=300)
