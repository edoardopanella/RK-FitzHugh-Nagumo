import numpy as np
import matplotlib.pyplot as plt
from fixed_points import fixed_points
from fhn_model import RK4

# Global parameters (Sherwood 2013)
a = 0.7
b = 0.8
phi = 0.08

'''
Convergence

In the code below we will run convergence tests on two setup:

1. Resting regime (I=0)
2. Spiking regime (I = 0.5)

For both setups we will test the following stepsizes:
- 1/8
- 1/16
- 1/32
- 1/64

'''
# Define grid of step sizes
hs = [1/8, 1/16, 1/32, 1/64]

# ==============
# RESTING REGIME
# ==============

I_r = 0
t0 = 0
tN = 300

# Find fixed points
v_star_r, w_star_r, _ = fixed_points(a=a, b=b, I = I_r)

# Define initial condition. We take v_stim (the external stimulus)
# to be = 0.8, similar to what is taken in Sherwood 2013
v0_r = v_star_r + 0.8
w0_r = w_star_r

# Compute "true solution"
from convergence import u_reference, compute_error

u_ref_r = u_reference(
    v0= v0_r,
    w0 = w0_r,
    t0 = t0,
    tN = tN,
    I = I_r
)[0]

# Iterate through step sizes in hs and display global error
print()
print("E(h) for resting regime")
print()
for h in hs:
    # generate the approximation
    u = RK4(
        t0 = t0,
        tN = tN,
        h = h,
        v0=v0_r,
        w0=w0_r,
        I = I_r
    )[0]
    
    error = compute_error(u=u, u_ref=u_ref_r)
    print(f"h = {h} | E(h) = {error}")


# ==============
# SPIKING REGIME
# ==============

I_s = 0.5
t0 = 0
tN = 2000

# Find fixed points
v_star_s, w_star_s, _ = fixed_points(a=a, b=b, I = I_s)

# Define initial conditions with the same external stimulus (0.8)
v0_s = v_star_s + 0.8
w0_s = w_star_s

# Computer "true solution"
u_ref_s = u_reference(
    v0 = v0_s,
    w0 = w0_s,
    t0 = t0,
    tN=tN,
    I=I_s
)[0]

# Iterate through step sizes in hs and display global error

print()
print("E(h) for spiking regime")
print()

for h in hs:
    # generate the approximation
    u = RK4(
        t0 = t0,
        tN = tN,
        h = h,
        v0=v0_s,
        w0=w0_s,
        I = I_s
    )[0]
    
    error = compute_error(u=u, u_ref=u_ref_s)
    print(f"h = {h} | E(h) = {error}")


'''
Experiments

In the code below we will run a series of experiment that will consist in plotting phase planes and time
series of v(t) with different initial conditions and in both resting and tonic spiking regime. The ideas are
inspired by the original results from Fitzhugh 1961 and the detailed walkthrough from Sherwood 2013

The experiments are fully implemented in experiments.py, in this script we will simply call the functions in order
'''

from experiments import experiment_1, experiment_2, experiment_3, experiment_4

# Experiment 1: Single action potential from resting regime
experiment_1()

# Experiment 2: Dynamics under varying external stimuli
experiment_2()

# Experiment 3: Dynamics in the Tonic Spiking Regime
experiment_3()

# Experiment 4: Dynamics under varying external stimuli (I = 0.5)
experiment_4()

