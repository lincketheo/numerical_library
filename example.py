from ode import *
import matplotlib.pyplot as plt
from numpy import pi, sqrt



# An example application of ODE
if __name__ == '__main__':

    # The drive frequency
    alpha = 90
    # Create a simple pendulum system
    p = pendulum(beta = 0, l = 0.1, m = 0.1, g = 9.8, A = 0, alpha = 0)

    # Create a new ode with the pendulum state function 
    a = ODE(state_func = p.state_func, size=2, start_state = [0.1, 0], time_step=0.0001)

    # Create a temporal slice 
    states_osc, states= a.temporal_slice(n = 10, T = 1 / p.natfreq, reset_on_end = False, verbose = True, max_states = 100000000, ode_states=True)
    
    # Plot the temporal slice on top of the trajectory
    plt.scatter(states[:,0], states[:,1], s = 0.01, label="Trajectory", c ="blue")
    plt.scatter(states_osc[:,0], states_osc[:,1], s = 10, c ="red")
    plt.title("Temporal Slice Plot Damped Driven, Temporal Period = Drive Period")
    plt.xlabel("Orientation (radians)")
    plt.ylabel("Angular Velocity (radians / s)")
    plt.legend()
    plt.show()

