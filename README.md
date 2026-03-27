# information-engine
Code and analytical solutions for the optimal control of a mesoscopic information engine. Evaluates finite-time transport of an overdamped particle in an optical trap under costly measurement via a POMDP framework. Maps exact thermodynamic boundaries for discrete and continuous-measurement Maxwell demons.

# Optimal Control of a Mesoscopic Information Engine

This repository contains the exact analytical solutions and simulation code required to reproduce the figures and thermodynamic boundaries presented in the manuscript *"Optimal Control of a Mesoscopic Information Engine"*. 

The code evaluates the finite-time control problem of driving an overdamped particle via an optical trap under costly measurement, formulated as a Partially Observable Markov Decision Process (POMDP).

## Dependencies
The scripts are written in standard Python 3 and require the following packages:
* `numpy` (Numerical arrays and stochastic sampling)
* `scipy` (Specifically `scipy.optimize` for exact steady-state limits)
* `matplotlib` 

## Repository Structure

The repository is organized into four standalone scripts corresponding to the manuscript's figures:

* **`fig1_binary_trajectory.py`**
  Simulates the finite-time spatial transport using a binary perfect sensor. It evaluates the 1D discrete Riccati recurrence and dynamic programming thresholds to output the true particle position, belief state, trap position, and finite-time "deadline blindness" trigger.

* **`fig2_binary_phasespace.py`**
  Maps the macroscopic steady-state thermodynamics of the binary sensor. Computes the optimal measurement frequency and plots the exact theoretical starvation envelope $C_{env}(v)$ that separates the active engine from the net-dissipative drag regime.

* **`fig3_continuous_trajectory.py`**
  Simulates the optimal closed-loop transport using a variable-precision (Kalman) continuous sensor. Computes the true global DP solution via value iteration and compares the dynamically optimal variance target against the steady-state algebraic limits.

* **`fig4_continuous_phasespace.py`**
  Evaluates the exact steady-state precision limits of the continuous sensor. Solves the resulting cubic polynomial via Cardano's formula to map the precision effort and plot the viability envelope $c_{env}(v)$.

## Execution
Each script executes independently and evaluates the analytical physics without requiring external datasets. Run the files directly from the command line:

```bash
python fig1_binary_trajectory.py
