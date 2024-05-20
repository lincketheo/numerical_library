from ode import *
import matplotlib.pyplot as plt
from numpy import pi, sqrt

# An example application of ODE
if __name__ == '__main__':

    # The drive frequency
    alpha = 90


    # A simple pendulum being driven by a drive frequency equal to 90 HZ
    # With drive frequency amplitude = A
    p = pendulum(beta = 0.01, l = 0.1, m = 0.1, g = 9.8, A = 1, alpha = alpha)

    # Create a new ode with the pendulum state function 
    a = ODE(state_func = p.state_func, size=2, start_state = [3, 2.5], time_step=0.001)
    t, states = a.ode(10000)
    plt.plot(states[0], states[1])
    plt.show()


    # THESE USUALLY TAKE A LOT OF TIME - Creates a temporal slice

    # Create a temporal slice 
    states_osc = a.temporal_slice(n = 1000, T = np.pi * 2 / alpha , reset_on_end = False, verbose = True, max_states = 100000000)
    
    # Plot the temporal slice on top of the trajectory
    #  plt.scatter(states[:,0], states[:,1], s = 0.00001, label="Trajectory", c ="blue")
    plt.scatter(states_osc[:,0] % (2 * np.pi), states_osc[:,1], s = 1, label="Temporal Slice", c ="red")
    plt.title("Temporal Slice Plot With Linear Interpolation, Temporal Period = Drive Period")
    plt.xlabel("Orientation (radians)")
    plt.ylabel("Angular Velocity (radians / s)")
    plt.legend()
    plt.show()
