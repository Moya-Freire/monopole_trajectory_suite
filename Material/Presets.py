# Presets are dictionaries with the mass fractions of each element in a given medium,
# used by the material class to estimate the resulting stopping power for it.

air = {
    "N": 0.755,
    "O": 0.232,
    "Ar": 0.013,
}

# Bulk crust, mass fractions. CRC Handbook crustal abundances, renormalised
# over the elements present in ATOMS (sums to 1).
crust = {
    "O":  0.460722,
    "Si": 0.281830,
    "Al": 0.082250,
    "Fe": 0.056266,
    "Ca": 0.041475,
    "Na": 0.023586,
    "Mg": 0.023286,
    "K":  0.020887,
    "Ti": 0.005647,
    "H":  0.001399,
    "P":  0.001049,
    "Mn": 0.000949,
    "S":  0.000350,
    "C":  0.000200,
    "Ni": 0.000084,
    "N":  0.000019,
}

water = {
    "H": 0.111894,
    "O": 0.888106,
}

ice = {
    "H": 0.111894,
    "O": 0.888106,
}

# Standard ocean water, approximately 35 g/kg salinity.
# Elemental mass fractions normalized to sum to 1.
# Trace elements not included in the reference composition are omitted.

ocean_water = {
    "H":  0.108211146,
    "O":  0.858488424,
    "Cl": 0.019401998,
    "Na": 0.010801113,
    "Mg": 0.001292133,
    "S":  0.000910094,
    "Ca": 0.000400041,
    "K":  0.000400041,
    "Br": 0.000067007,
    "C":  0.000028003,
}

# Lithospheric mantle, mass fractions. Moderately depleted peridotite
# (melt-extraction residue), anchored on a garnet-derived lherzolite
# composition from Griffin et al. 2003: SiO2 44.4, MgO 43.2, FeO 7.8,
# Al2O3 1.9, CaO 1.6, Na2O 0.13, NiO 0.30, MnO 0.13, TiO2 0.07 wt%.
# Oxides converted to elemental mass fractions, renormalised over the
# elements present in ATOMS (sums to 1).

lithospheric_mantle = {
    "O":  0.443180,
    "Mg": 0.261404,
    "Si": 0.208251,
    "Fe": 0.060837,
    "Ca": 0.011474,
    "Al": 0.010090,
    "Ni": 0.002365,
    "Mn": 0.001010,
    "Na": 0.000968,
    "Ti": 0.000421,
}


# When adding a new preset, add the dictionary
# above and its name in the preset dictionary below to make it usable
presets = {"Air": air,
           "Crust": crust,
           "Water":water,
           "Ice":ice,
           "OceanWater":ocean_water,
           "UpperMantle":lithospheric_mantle,
           "Vacuum":None,
}
