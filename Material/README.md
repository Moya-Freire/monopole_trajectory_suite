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

$$\frac{dE}{dx}=
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

### Bartlett et al.: Monopole stopping power in air

The stopping-power parameterization used for air is taken from Equation (3), page 6, of:

- D. F. Bartlett, D. Soo, R. L. Fleischer, H. R. Hart Jr., and A. Mogro-Campero, *Search for Cosmic-Ray-Related Magnetic Monopoles at Ground Level*, FERMILAB-PUB-81-103-E (April 1981).
- [Read the original paper (PDF)](https://lss.fnal.gov/archive/1981/pub/fermilab-pub-81-103-e.pdf)

The energy loss per unit areal density is given by

```math
\zeta(\beta) =
\begin{cases}
1.55 n^2 \left[\ln(\gamma\beta)+4\right],
& \beta > 0.03, \\[6pt]
860 n^2 \beta^2,
& \beta \leq 0.03,
\end{cases}
```

where

```math
\gamma = \frac{1}{\sqrt{1-\beta^2}}.
```

Here, $n$ is the monopole's magnetic charge in units of the Dirac charge. The stopping power is expressed in $\mathrm{GeV}/(\mathrm{g\,cm^{-2}})$.

Implemented in `Material.bartlett()` and selected by `Material.getStopPower()` for materials whose preset is `"Air"`. The implementation converts the stopping power to energy loss per unit length using the material density.

### Ahlen: Bethe–Bloch stopping power for magnetic monopoles

The high-velocity monopole stopping-power calculation follows the Bethe–Bloch formalism adapted for magnetic charges, including the relevant quantum, Bloch, and density-effect corrections.

- S. P. Ahlen, *Stopping-power formula for magnetic monopoles*, Physical Review D **17**, 229–233 (1978).
- [https://doi.org/10.1103/PhysRevD.17.229](https://doi.org/10.1103/PhysRevD.17.229)

The implemented expression is

```math
\frac{dE}{dx}
=
\frac{\pi\hbar^2 N_e n^2}{m_e}
\left[
\ln\left(\frac{2m_ec^2\beta^2\gamma^2}{I}\right)
+\frac{K(n)}{2}
-\frac{\delta}{2}
-\frac{1}{2}
-B(n)
\right]
```

where $N_e$ is the electron number density, $n$ is the magnetic charge in units of the Dirac charge, $I$ is the mean excitation energy, $\delta$ is the density-effect correction, and $K(n)$ and $B(n)$ account for monopole-specific corrections.

Implemented in `Material.bethe()` and used by `Material.getStopPower()` for $\beta > 0.1$.
