import numpy as np
import matplotlib.pyplot as plt
from fixed_points import fixed_points

# Parameters
a = 0.7
b = 0.8 
phi = 0.08

# Vector field defining the system (Sherwood 2013)
def F(u, I):
    v = u[0]
    w = u[1]
    x = v - (v**3)/3 - w + I
    y = phi * (v + a - b*w)
    return np.array([x, y])

# Implementation of Runge-Kutta of order 4
def RK4(t0, tN, h, v0, w0, I):

    N = int((tN - t0)/h)
    u = np.array([v0, w0])
    t = t0

    us = [u]
    ts = [t0]

    for i in range(N):
        K1 = F(u, I)
        K2 = F(u + h*(K1/2), I)
        K3 = F(u + h*(K2/2), I)
        K4 = F(u + h*K3, I)

        u = u + (h/6)*(K1 + 2*K2 + 2*K3 + K4)
        t = t+h

        us.append(u)
        ts.append(t)
    return np.array(us), np.array(ts)

# Nullclines
def f_v(v, I):
    return v - (v**3)/3 + I

def f_w(v, a, b):
    return (v+a)/b