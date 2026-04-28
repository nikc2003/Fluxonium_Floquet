# Fluxonium branch analysis package

meant to implement a **capacitively coupled fluxonium-resonator model** and a **recursive dressed-state branch analysis** adapted from the transmon branch-labeling scheme to fluxonium (fully based on Dumas et. al https://doi.org/10.1103/PhysRevX.14.041023).

## Rundown of this fodler
- fluxonium diagonalization in the harmonic-oscillator basis
- construction of resonator with capacitive coupling through the fluxonium charge operator
- undriven dressed-state diagonalization
- branch construction by recursive `a^dag` overlaps
- also diagnostic checks
  - average bare fluxonium level
  - average resonator photon number
  - computational subspace population
  - reduced-fluxonium purity
  - fluxonium participation ratio
- also driven Hamiltonian and Lindblad helpers 

## layout
- `fluxonium_branch/fluxonium_model.py` — bare fluxonium
- `fluxonium_branch/composite_system.py` — coupled  system
- `fluxonium_branch/branch_analysis.py` — branch assignment
- `fluxonium_branch/diagnostics.py` — state/branch diagnostics
- `fluxonium_branch/time_evolution.py` — for the direct driven simulations
- `fluxonium_branch/plotting.py` — basic plotting helpers
- `test.py` —  script using aggron qA params

## Hamiltonian convention

The bare fluxonium Hamiltonian is
`H = 4 E_C n^2 + (1/2) E_L phi^2 - E_J cos(phi + 2*pi*flux)` (followinf scqubits)
The coupled undriven Hamiltonian is
`H0 = omega_r a^dag a + H_fluxonium - i g_qr n (a - a^dag)`.

## important note on units
I jsut use ordinary freq units.

- `EJ_GHz`, `EC_GHz`, `EL_GHz`, `omega_r_GHz`, `g_qr_GHz`, `epsilon_d_GHz`, `omega_d_GHz`
- `kappa_MHz`

But internally, the code converts everything to angular-frequency units.

## to start

```
in the terminal:

python test.py
```
This will:
1. build the fluxonium for aggron qA cfg
2. print the lowest few transition frequencies and charge matrix elements
3. build the coupled fluxonium-resonator Hamiltonian
4. diagonalize it
5. run branch analysis
6. save diagnostic plots for branches seeded by bare levels 0 and 1

## Notes
- The branch analysis itself uses the **undriven** Hamiltonian!! iteration done through simply adding photons to the resoantor subspace 
- there are some preliminary driven Lindblad helpers for direct time-domain simulations  (if want to go into what Google IST did), but they are not required for building branches. 
  I also haven't rigoruously tested if this part works in full yet.
