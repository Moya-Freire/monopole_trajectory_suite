# Geometry module
In order for the propagator to work, it must be told what the particle is being propagated through. The *Primitives.py* file encodes different 3D shapes,
and the *SimulationVolume* class, defined in *SimulationVolume.py*, provides us with the means to arrange them in whatever the ensemble we want to study,
as well as wraps the queries for the density and stopping power of a given medium in a convenient way.
