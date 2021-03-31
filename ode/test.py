from numerical_lib import *
import numpy as np
import matplotlib.pyplot as plt



if __name__ == '__main__':
    f = lambda y, t : y
    fsltn = lambda t : np.exp(t)

    tol = 0.01
    t = np.arange(0, 2, 0.01)
    e = eulers(f, 0.1, 0, 2, 1, tollerance = tol)
    m = midpoint(f, 0.1, 0, 2, 1, tollerance = tol)
    r = rk4(f, 0.1, 0, 2, 1, tollerance = tol)
    a = fsltn(t)

    plt.plot(list(zip(*e))[0], list(zip(*e))[1], '--go')
    plt.plot(list(zip(*m))[0], list(zip(*m))[1], '--bo')
    plt.plot(list(zip(*r))[0], list(zip(*r))[1], '--ro')
    plt.plot(t, a)

    plt.show()
