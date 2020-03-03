from ode import *
import matplotlib.pyplot as plt
from numpy import pi, sqrt

# An example application of ODE
if __name__ == '__main__':


    l = lorentz(a = 16.0, r = 45.0, b = 4.0)

    start = [0.0, -1.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

    a = ODE(state_func = l.variational_state_func, \
            size = 12, start_state = start, \
            time_step = 0.001)


    time, states = a.ode(10000, adaptive_t = False, \
                         reset_on_end = True, \
                         verbose = False)
   
    print(states.shape)
    print(states[:,states.shape[1] - 1][-9:].reshape(3, 3))
