import numpy as np
import matplotlib.pyplot as plt
from fixed_points import fixed_points
from fhn_model import RK4, f_v, f_w

# Plot settings

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300


plt.rcParams['font.family'] = 'serif' 
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'cm' 
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 11

# 3. Line Weights
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['axes.linewidth'] = 1.2

# Global parameters (Sherwood 2013)

a = 0.7
b = 0.8
phi = 0.08

def run_from_fixed_point(I, v_stim, t0, tN, h=0.03125):
    '''
    Integrate FHN from (v0 + v_stim, w0) with current I
    '''
    v_star, w_star, _ = fixed_points(a=a, b=b, I=I)

    # Initial conditions
    v0 = v_star + v_stim
    w0 = w_star

    us, ts = RK4(t0=t0, tN=tN, h=h, v0=v0, w0=w0, I=I)

    return us, ts, v_star, w_star, v0, w0

def plot_grid(I, ax):
    v_grid = np.linspace(-2.3, 2.3, 25) 
    w_grid = np.linspace(-1.1, 1.4, 20)
    V, W = np.meshgrid(v_grid, w_grid)

    DV = V - (V**3)/3 - W + I
    DW = phi * (V + a - b*W)

    M = np.hypot(DV, DW)
    M[M == 0] = 1  
    DV_norm = DV / M
    DW_norm = DW / M

    ax.quiver(V, W, DV_norm, DW_norm, 
          pivot='mid', 
          color='#555555',  # Dark grey
          alpha=0.3,        # Slight transparency
          width=0.003, 
          scale=30)         # Adjust scale to change arrow length


def experiment_1():
    '''
    Single action potential ignited by an external stimu
    '''

    # Define time as in Sherwood 2013
    t0, tN = 0, 300
    I=0
    v_stim = 0.8

    us, ts, v_star, w_star, v0, w0 = run_from_fixed_point(I=I, v_stim=v_stim, t0=t0, tN=tN)

    vs = us[:, 0]
    ws = us[:, 1]

    # Plot

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot trajectory
    ax.plot(vs, ws, color='black', linewidth=2.0 ,label='Trajectory')

    # Plot fixed point
    ax.scatter(v_star, w_star, s=100, color= 'blue', zorder= 10,label= 'Fixed point $(v^*_R, w^*_R)$')

    # Plot initial condition
    ax.scatter(v0, w0, marker='x', color='purple', s=100, zorder = 10,linewidth =2.5,label='Initial Condition $(v^*_R + v_s, w^*_R)$')

    # Plot nullclines
    vs_ref = np.linspace(-2.2, 2.2)
    ax.plot(vs_ref, f_v(vs_ref, I), color='red', label='v nullcline', linestyle = '--', linewidth = 1.5)
    ax.plot(vs_ref, f_w(vs_ref, a, b), color='green', label='w nullcline', linestyle = '--', linewidth = 1.5)

    ax.set_xlabel('v (voltage)')
    ax.set_ylabel('w (recovery)')

    ax.grid()
    ax.legend()
    ax_ins = ax.inset_axes([0.6, 0.7, 0.35, 0.25])
    
    ax_ins.plot(ts, vs, color='black', linewidth=1.2)
    ax_ins.set_xlabel("Time", fontsize=9)
    ax_ins.set_ylabel("v(t)", fontsize=9)

    ax_ins.set_xlim(0, 300)
    ax_ins.set_ylim(-2.2, 2.2)
    ax_ins.grid(True, linestyle=':', alpha=0.5)

    ax_ins.patch.set_alpha(0.9)
    ax.set_title("Single Action Potential Triggered from the Resting Fixed Point")
    fig.set_dpi(200)
    fig.tight_layout()
    plt.savefig('plots/experiment_1.png')
    plt.show()


def experiment_2():

    '''
    Dynamics under varying external stimuli
    '''

    # Define time as in Sherwood 2013
    t0, tN = 0, 300
    I=0

    fig, ax = plt.subplots(figsize=(9, 7))
    ax_ins = ax.inset_axes([0.6, 0.7, 0.45, 0.35])
    ax.set_xlabel('v (voltage)')
    ax.set_ylabel('w (recovery)')
    ax_ins.set_xlabel("Time", fontsize=9)
    ax_ins.set_ylabel("v(t)", fontsize=9)

    # Plot nullclines
    vs_ref = np.linspace(-2.2, 2.2)
    ax.plot(vs_ref, f_v(vs_ref, I), color='red', label='v nullcline', linestyle = '--', linewidth = 1.5)
    ax.plot(vs_ref, f_w(vs_ref, a, b), color='green', label='w nullcline', linestyle = '--', linewidth = 1.5)

    # Define a range of stimuli that enable a clear overview of the dynamic
    stimuli_v = [0.4, 0.5, 0.54 , 0.55, 0.555, 0.5555, 0.56, 0.7]
    colors = plt.cm.cool(np.linspace(0, 1, 8))


    for i in range(len(stimuli_v)):
        v_stim = stimuli_v[i]
        us, ts, v_star, w_star, v0, w0 = run_from_fixed_point(I=I, v_stim=v_stim, t0=t0, tN=tN)
        vs = us[:, 0]
        ws = us[:, 1]

        ax.scatter(v0, w0, marker='x', color=colors[i], s=50, zorder = 10,linewidth =2.5, alpha=0.5)
        ax.plot(vs, ws, color=colors[i], linewidth=2.0 ,label=f'Stimulus = {v_stim}', alpha=0.7)
        ax_ins.plot(ts, vs, color=colors[i], linewidth=1.2)

    # Plot fixed point
    ax.scatter(v_star, w_star, s=100, color= 'blue', zorder= 10,label= 'Fixed point $(v^*_R, w^*_R)$')


    ax_ins.set_xlim(0, 300)
    ax_ins.set_ylim(-2.2, 2.2)
    ax_ins.grid(True, linestyle=':', alpha=0.5)

    ax_ins.patch.set_alpha(0.9)

    # Plot arrows grid
    plot_grid(I=I, ax=ax)


    ax.grid()
    ax.legend()
    ax.set_title("Phase-Plane Responses to Varying Initial Conditions $(I=0)$")

    fig.set_dpi(200)
    fig.tight_layout()
    plt.savefig('plots/experiment_2.png')
    plt.show()

def experiment_3():
    '''
    Tonic spiking regime (I=0.5)
    '''
    # Define time as in Sherwood 2013
    t0, tN = 0, 2000
    I=0.5
    v_stim = 0.8

    us, ts, v_star, w_star, v0, w0 = run_from_fixed_point(I=I, v_stim=v_stim, t0=t0, tN=tN)
    vs = us[:, 0]
    ws = us[:, 1]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot old nullclines
    vs_ref = np.linspace(-2.2, 2.2)
    ax.plot(vs_ref, f_v(vs_ref, I=0), color='red', label='v nullcline (I=0)', linestyle= ':', linewidth = 1.5)


    # Plot new nullclines
    ax.plot(vs_ref, f_v(vs_ref, I=I), color='red', label='v nullcline (I=0.5)', linestyle= '--', linewidth = 1.5)
    ax.plot(vs_ref, f_w(vs_ref, a, b), color='green', label='w nullcline', linestyle= '--', linewidth = 1.5)

    # Plot trajectory
    ax.plot(vs, ws, color='black', linewidth=2.0 ,label='Trajectory')

    # Plot fixed point
    ax.scatter(v_star, w_star, s=100, color= 'blue', zorder= 10,label= 'Fixed point $(v^*_S, w^*_S)$')

    # Plot initial condition
    ax.scatter(v0, w0, marker='x', color='purple', s=100, zorder = 10,linewidth =2.5,label='Initial Condition $(v^*_S + v_s, w^*_S)$')

    ax.set_xlabel('v (voltage)')
    ax.set_ylabel('w (recovery)')

    ax.grid()
    ax.legend()
    ax_ins = ax.inset_axes([0.55, 0.7, 0.55, 0.25])

    ax_ins.plot(ts, vs, color='black', linewidth=1.2)
    ax_ins.set_xlabel("Time", fontsize=9)
    ax_ins.set_ylabel("v(t)", fontsize=9)

    ax_ins.set_xlim(0, 2000)
    ax_ins.set_ylim(-2.2, 2.2)
    ax_ins.grid(True, linestyle=':', alpha=0.5)

    ax_ins.patch.set_alpha(0.9)
    ax.set_title("Tonic Spiking Regime")
    fig.set_dpi(200)
    fig.tight_layout()
    plt.savefig('plots/experiment_3.png')
    plt.show()

def experiment_4():

    '''
    Dynamics under varying external stimuli (I=0.5)
    '''

    # Define time as in Sherwood 2013
    t0, tN = 0, 2000
    I=0.5

    fig, ax = plt.subplots(figsize=(9, 7))
    ax_ins = ax.inset_axes([0.6, 0.7, 0.45, 0.35])
    ax.set_xlabel('v (voltage)')
    ax.set_ylabel('w (recovery)')
    ax_ins.set_xlabel("Time", fontsize=9)
    ax_ins.set_ylabel("v(t)", fontsize=9)

    # Plot nullclines
    vs_ref = np.linspace(-2.2, 2.2)
    ax.plot(vs_ref, f_v(vs_ref, I=0), color='red', label='v nullcline (I=0)', linestyle = ':', linewidth = 1.5)
    ax.plot(vs_ref, f_v(vs_ref, I), color='red', label='v nullcline', linestyle = '--', linewidth = 1.5)
    ax.plot(vs_ref, f_w(vs_ref, a, b), color='green', label='w nullcline', linestyle = '--', linewidth = 1.5)

    # Define a range of stimuli that enable a clear overview of the dynamic
    stimuli_v = [0.2, 0.3, 0.4, 0.55, 0.555, 0.5555, 0.56, 0.7]
    colors = plt.cm.cool(np.linspace(0, 1, 8))


    for i in range(len(stimuli_v)):
        v_stim = stimuli_v[i]
        us, ts, v_star, w_star, v0, w0 = run_from_fixed_point(I=I, v_stim=v_stim, t0=t0, tN=tN)
        vs = us[:, 0]
        ws = us[:, 1]

        ax.scatter(v0, w0, marker='x', color=colors[i], s=50, zorder = 10,linewidth =2.5, alpha=0.5)
        ax.plot(vs, ws, color=colors[i], linewidth=2.0 ,label=f'Stimulus = {v_stim}', alpha=0.7)
        ax_ins.plot(ts, vs, color=colors[i], linewidth=1.2)

    # Plot fixed point
    ax.scatter(v_star, w_star, s=100, color= 'blue', zorder= 10,label= 'Fixed point $(v^*_S, w^*_S)$')


    ax_ins.set_xlim(0, 2000)
    ax_ins.set_ylim(-2.2, 2.2)
    ax_ins.grid(True, linestyle=':', alpha=0.5)

    ax_ins.patch.set_alpha(0.9)

    # Plot arrows grid
    plot_grid(I=I, ax=ax)


    ax.grid()
    ax.legend()
    ax.set_title("Phase-Plane Responses to Varying Initial Conditions $(I=0.5)$")

    fig.set_dpi(200)
    fig.tight_layout()
    plt.savefig('plots/experiment_4.png')
    plt.show()
