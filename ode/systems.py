import numpy as np

# A useful file for defining some nonlinear system functions

##
# @brief Rossler system of equations
class rossler:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def state_func(self, state, t):
        xdot = -(state[1] + state[2])
        ydot = state[0] + a * state[1]
        zdot = b + state[2] * (state[0] - c)
        return np.array([xdot, ydot, zdot])


##
# @brief Lorentz system of equations
class lorentz:
    def __init__(self, a, r, b):
        self.a = a
        self.r = r
        self.b = b

    def state_func(self, state, t):
        xdot = self.a * (state[1] - state[0])
        ydot = self.r * state[0] - state[0] - state[0] * state[2]
        zdot = state[0] * state[1] - self.b * state[2]
        return np.array([xdot, ydot, zdot])

##
# @brief Pendulum system of equations
class pendulum:
    def __init__(self, beta, l, m, g, A, alpha):
        self.beta = beta
        self.l = l
        self.m = m
        self.g = g
        self.A = A
        self.alpha = alpha
        self.natfreq = np.sqrt(self.g / self.l) / (2 * np.pi)

    def state_func(self, state, t):
        thetadot = state[1]
        omegadot = (self.A * np.cos(self.alpha * t) - self.beta * self.l * state[1] - self.m * self.g * np.sin(state[0])) / (self.m * self.l)
        return np.array([thetadot, omegadot])
