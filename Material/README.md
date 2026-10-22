# Stopping power calculator and composite material suite

This code, based on extensive literature reviews of monopole stopping power in different media, provides tools to estimate the stopping power a monopole experiences while traversing a given medium, based on the relative abundances of its constituent elements.

## Sources

### Giacomelli: Very-low-velocity monopole stopping power

- G. Giacomelli, *Magnetic Monopoles Searches*, DFUB 20/94, invited lectures at the 1994 Lake Louise Winter Institute.
- Section 3.3, Equation (11), page 8: approximate stopping power due to elastic monopole–atom collisions for $\beta < 10^{-4}$.

$$
\frac{dE}{dx} \simeq N_a \frac{\hbar^2}{m_e}
$$

- Implemented in `Material.giacomelli()` and selected by `Material.getStopPower()` for $\beta < 10^{-4}$.

[Read the original paper (PDF)](https://lss.fnal.gov/archive/other/dfub-20-94.pdf)

### Ahlen–Kinoshita: Low-velocity monopole stopping power

Used to calculate the electronic stopping power of slow, noncatalyzing magnetic monopoles in condensed materials.

The implementation follows the stopping-power expression given in Section II, Equations (1) and (2), of:

- M. A. Acero et al. (NOvA Collaboration), *Search for slow magnetic monopoles with the NOvA detector on the surface*, Physical Review D **103**, 012007 (2021).
- [https://doi.org/10.1103/PhysRevD.103.012007](https://doi.org/10.1103/PhysRevD.103.012007)

The expression is

$$
\frac{dE}{dx}
=
aN_e^{2/3}
\left[
\ln\left(bN_e^{1/3}\right)-\frac{1}{2}
\right]\beta
$$

where $N_e$ is the material's electron number density, and

$$
a = \frac{2\pi g^2 e^2}
{\hbar c(3\pi^2)^{1/3}},
\qquad
b = 2(3\pi^2)^{1/3}a_0
$$

The paper attributes this calculation to the original reference:

- S. P. Ahlen and K. Kinoshita, *Calculation of the stopping power of very-low-velocity magnetic monopoles*, Physical Review D **26**, 2347 (1982).

Implemented in `ahlen_kinoshita()` and called by `Material.getStopPower()` for $10^{-4} \leq \beta < 10^{-2}$.
