import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class LinearEqu:

    def __init__(self, state_func, size, tolerance = 100000, start_state = None, time_step = 0.001):
        self.size = size
        self.func = state_func
        self.h = time_step
        self.time = 0.0
        self.tolerance = tolerance

        if start_state == None or len(start_state) != self.size:
            self.state = np.zeros(self.size)
        else:
            self.state = np.array(start_state)

        
    def de_step_rk4(self, h, t, x):
        k1 = h * self.func(state = x, t = t)
        k2 = h * self.func(state = x + k1 / 2, t = t + h / 2)
        k3 = h * self.func(state = x + k2 / 2, t = t + h / 2)
        k4 = h * self.func(state = x + k3, t = t + h)

        return t + h, (x + (k1 + 2 * k2 + 2 * k3 + k4) / 6)

    def step(self, adaptive_t):
        if adaptive_t:

            error = self.tolerance + 1
            i = 10
            
            # One step forward
            t1, x1 = self.de_step_rk4(self.h, self.time, self.state)

            # Two steps forward
            t2, x2 = self.de_step_rk4(self.h / 2, self.time, self.state)
            t3, x3 = self.de_step_rk4(self.h / 2, t2, x2)

            error = np.linalg.norm(x3 - x1)

            # This could be done with recursion,
            # But recursion is evil
    

            if(error > self.tolerance):
                while error > self.tolerance and i > 0:
                    self.h = self.h / 2

                    # One step forward
                    t1, x1 = self.de_step_rk4(self.h, self.time, self.state)

                    # Two steps forward
                    t2, x2 = self.de_step_rk4(self.h / 2, self.time, self.state)
                    t3, x3 = self.de_step_rk4(self.h / 2, t2, x2)

                    error = np.linalg.norm(x3 - x1)
                    i -= 1

                if i <= 0:
                    print("Greater Than Exceeded depth")

                # Pick the best step no matter what
                self.time, self.state = t3, x3

            elif(error < self.tolerance):

                # Pick the best step
                self.time, self.state = t3, x3

                while error < self.tolerance and i > 0:
                    self.h = self.h * 2

                    # One step forward
                    t1, x1 = self.de_step_rk4(self.h, self.time, self.state)

                    # Two steps forward
                    t2, x2 = self.de_step_rk4(self.h / 2, self.time, self.state)
                    t3, x3 = self.de_step_rk4(self.h / 2, t2, x2)

                    error = np.linalg.norm(x3 - x1)
                    i -= 1

                if i <= 0:
                    print("Less Than Exceeded depth")

            return
        else:
            t1, x1 = self.de_step_rk4(self.h, self.time, self.state)
            self.time, self.state = t1, x1
            return

    def ode(self, n, adaptive_t = False):
        time = np.zeros(n)
        states = np.zeros(n * self.size).reshape(self.size, n)

        for i in range(n):
            self.step(adaptive_t)
            time[i] = self.time
            states[:,i] = self.state

        return time, states

    def reset(self, start_state = None):
        if start_state == None or len(start_state) != self.size:
            self.state = np.zeros(self.size)
        else:
            self.state = np.array(start_state)

        self.time = 0.0

a = 16
r = 900
b = 4


#  a = 0.398
#  b = 2
#  c = 4

def func(state, t):
    xdot = a * (state[1] - state[0])
    ydot = r * state[0] - state[0] - state[0] * state[2]
    zdot = state[0] * state[1] - b * state[2]
    return np.array([xdot, ydot, zdot])

#  def func_rossler(state, t):
#      xdot = -(state[1] + state[2])
#      ydot = state[0] + a * state[1]
#      zdot = b + state[2] * (state[0] - c)
#      return np.array([xdot, ydot, zdot])


if __name__ == '__main__':
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #  start = [-13, -12, 52]
    start = [100, 100, 100]

    lorentz = LinearEqu(state_func = func, size = 3, tolerance = 0.0001, start_state = start, time_step = 1)
    t, states = lorentz.ode(10000, True)
    #  lorentz.reset([-13, -12, 52])
    #  lorentz.h = 0.001
    #  t, states_2 = lorentz.ode(50, False)


    x = states[0]
    y = states[1]
    z = states[2]

    #  x_2 = states_2[0]
    #  y_2 = states_2[1]
    #  z_2 = states_2[2]
    #
    #  ax.scatter(x_2, y_2, z_2, zdir='z', c='blue', s=10, label="Stationary T Step")
    ax.scatter(x, y, z, zdir='z', c='black', label="Adaptive T Step", s=1)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f'Rossler Adaptive Time Step')
    plt.legend()
    plt.show()
