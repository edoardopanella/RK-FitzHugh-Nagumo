import numpy as np
import matplotlib.pyplot as plt
from fhn_model import RK4

def u_reference(v0, w0, t0, tN, I, h_ref = 1/1024):
    '''
    Wrapper function to compute the "true solution"
    '''

    return RK4(t0=t0, tN=tN, h=h_ref, v0=v0, w0=w0, I=I)

def compute_error(u, u_ref):
    '''
    Compute the max global error between an approximant and the "true solution"
    '''
    
    # Compute the step so that we can successfully filter u_ref for the common points
    step = (u_ref.shape[0]-1) // (u.shape[0]-1)
    
    # Filer u_ref with the step
    u_ref_filtered = u_ref[::step]

    diff = u - u_ref_filtered
    point_wise_norms = np.linalg.norm(diff, axis=1)

    return np.max(point_wise_norms)

