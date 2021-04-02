import numpy as np
import matplotlib.pyplot as plt

from ode.numerical_lib import *
from ode.systems import pendulum


if __name__ == '__main__':

    p = pendulum(beta = 0, l = 1, m = 1, g = 1, A = 0, alpha = 0)

    t, y = rk4(p.state_func, 0.01, 0, 10, np.array([0.1, 0.1]))
    ta, ya = rk4(p.state_func, 0.01, 0, 10, np.array([0.1, 0.1]), tollerance = 0.0001)
    
    states = np.array(y).T
    statesa = np.array(ya).T

    #  plt.scatter(states[0], states[1], s = 1)
    plt.scatter(statesa[0], statesa[1], s = 1)
    plt.show()
