import numpy as np
import matplotlib.pyplot as plt


# Simple linear equation        
class LinearEqu:

    def __init__(self, state_func, size, tolerance = 0, start_state = None):

        self.size = size


        # State transition function
        self.func = state_func


        # Update starting state
        if start_state == None or len(start_state) != self.size:
            self.state = np.zeros(self.size)
        else:
            self.state = np.array(start_state)

        # Preallocate space for next array
        self.next_state = np.zeros(self.size)

        # Time
        self.time = 0.0

    def de_step_rk4(self, h, t, x):
        k1 = h * self.func(state = x, t = t)
        k2 = h * self.func(state = x + k1 / 2, t = t + h / 2)
        k3 = h * self.func(state = x + k2 / 2, t = t + h / 2)
        k4 = h * self.func(state = x + k3, t = t + h)

        # Return new h, new state
        return t + h, (x + (k1 + 2 * k2 + 2 * k3 + k4) / 6)


    def step(self, h, adaptive_t):
        if adaptive_t:

            # h step
            t1, x1 = self.de_step_rk4(h, self.time, self.state)

            # h / 2 step
            t2, x2 = self.de_step_rk4(h / 2, self.time, self.state)
            t3, x3 = self.de_step_rk4(h / 2, t1, x2)

            error = np.linalg.norm(x3 - x1)

    
        



            t1, x1 = self.de_step_rk4(h, self.time)
            t1, 
            t1, x1 = self.de_step_rk4(h / 2, self.time)

            self.time, self.next_state = self.de_step_rk4(h, self.time)
            


        if not adaptive_t:
            # Update t, x_next
            self.time, self.next_state = self.de_step_rk4(h, self.time)
            # Swap next with current
            self.next_state, self.state = self.state, self.next_state


    def ode(self, h, n, adaptive_t = False):
        # An array of state vectors
        states = np.zeros(n * self.size).reshape(self.size, n)

        for i in range(n):
            self.step(h, adaptive_t)
            states[:,i] = self.state

        return states

    def reset(self, start_state = None):
        if start_state == None or len(start_state) != self.size:
            self.state = np.zeros(self.size)
        else:
            self.state = np.array(start_state)

        self.next_state = np.zeros(self.size)
        self.time = 0.0



def theta(state, t):
    return state[1]

def omega(state, t, beta, l, alpha, A, m, g):
    return (A * np.cos(alpha * t) - beta * l * state[1] - m * g * np.sin(state[0])) / (m * l)

def omega2(state, t, beta, l, alpha, A, m, g):
    return (A * np.cos(alpha * t) - beta * l * state[1] - m * g * state[0]) / (m * l)


def func(state, t):
    beta = 0
    l = 0.1
    m = 0.1
    g = 9.8

    natfreq = np.sqrt(g / l)

    A = 0
    alpha = 0

    return np.array([theta(state, t), \
            omega(state, t, beta = beta, l = l, alpha = alpha, A = A, m = m, g = g)])

def func2(state, t):
    beta = 0
    l = 0.1
    m = 0.1
    g = 9.8

    natfreq = np.sqrt(g / l)

    A = 0
    alpha = 0

    return np.array([theta(state, t), \
            omega(state, t, beta = beta, l = l, alpha = alpha, A = A, m = m, g = g)])



if __name__ == '__main__':

    states = [[0, 8.0], 
              [0, 17.0], 
              
              [np.pi - 0.0001, 0], 
              [np.pi + 0.0001, 0], 
              [-np.pi - 0.0001, 0], 

              [-2 * np.pi, 8.0],
              [-2 * np.pi, 17.0],

              [2 * np.pi, 8.0],
              [2 * np.pi, 17.0],

              [-7, 16.0],
              [7, -16.0],
              [-7, 30.0],
              [7, -30.0]]
              #  [-7, 50.0],
              #  [7, -50.0]]


    a = LinearEqu(func, 2, [0.0, 1.0])
    b = LinearEqu(func2, 2, [0.0, 1.0])
    #  output1 = a.ode(0.005, 3000, "rk4")
    #  output2 = b.ode(0.005, 3000, "rk4")


    #  output[0] = output[0]
    #  plt.plot(output[0], output[1], label="(Om_0, w_0) = (3, 0.1)")
    #  plt.xlabel("Theta (rad)")
    #  plt.ylabel("Omega (rad / s)")
    #  plt.title("State Space Trajectory Nonlinear Pendulum")
    #  plt.plot(output[0][0],output[1][0],'ro', label="Starting Point")
    #  plt.legend()


    for i in states:
        b.reset(i)
        a.reset(i)
        output1 = a.ode(0.005, 1000, "rk4")
        output2 = b.ode(0.005, 1000, "rk4")

        th1 = output1[0]
        om1 = output1[1]

        th2 = output2[0]
        om2 = output2[1]


        #  th = th % (2 * np.pi)

        plt.plot(th1, om1)
        plt.plot(th2, om2)



    plt.title("State Space Portrait Nonlinear Damped Pendulum Mod 2pi")
    plt.xlabel("Theta (rad)")
    plt.ylabel("Omega (rad / sec)")
    plt.xlim([-2 * np.pi, 2 * np.pi])
    plt.show()









