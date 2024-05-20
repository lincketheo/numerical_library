import numpy as np
import matplotlib.pyplot as plt

from ode.numerical_lib import *
from ode.systems import *


if __name__ == '__main__':
    p = pendulum(beta = 0, l = 1, m = 1, g = 1, A = 0, alpha = 0)

    t, y = rk4(p.state_func, 0.01, 0, 10, np.array([0.1, 0.1]))
    states = np.array(y).T
    plt.scatter(states[0], states[1], s = 1)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Pendulum RK4 without Adaptive Integrator")
    plt.show()

    ta, ya = rk4(p.state_func, 0.01, 0, 10, np.array([0.1, 0.1]), tollerance = 0.0001)
    statesa = np.array(ya).T
    plt.scatter(statesa[0], statesa[1], s = 1)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Pendulum RK4 with Adaptive Integrator")
    plt.show()

    r = rossler(0.2, 0.2, 5.7)

    ta, ya = rk4(r.state_func, 0.01, 0, 10000, np.array([0.1, 0.1, 0.1]), tollerance = 0.0001)
    statesa = np.array(ya).T
    plt.scatter(statesa[0], statesa[1], s = 0.1)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Rossler Slice RK4 with Adaptive Integrator")
    plt.show()


    l = lorentz(10, 28, 8/3)

    ta, ya = rk4(l.state_func, 0.01, 0, 1000, np.array([0.1, 0.1, 0.1]), tollerance = 0.0001)
    statesa = np.array(ya).T
    plt.scatter(statesa[0], statesa[1], s = 0.1)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Lorentz Slice RK4 with Adaptive Integrator")
    plt.show()
