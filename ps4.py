import numpy as np
import matplotlib.pyplot as plt



        


class LinearEqu:

    def __init__(self, state_func, size, start_state = None):

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

    def de_step_rk4(self, h, t):
        k1 = h * self.func(state = self.state, t = t)
        k2 = h * self.func(state = self.state + k1 / 2, t = t + h / 2)
        k3 = h * self.func(state = self.state + k2 / 2, t = t + h / 2)
        k4 = h * self.func(state = self.state + k3, t = t + h)

        self.next_state = self.state + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        self.time = self.time + h

    def de_euler(self, h, t):
        self.next_state = self.state + h * self.func(self.state, t)
            

    def step(self, h, func="rk4"):
        if(func == "rk4"):
            self.de_step_rk4(h, self.time)
            self.next_state, self.state = self.state, self.next_state
        elif(func == "euler"):
            self.de_euler(h, self.time)
            self.next_state, self.state = self.state, self.next_state
        else:
            print("unknown method")

    def ode(self, h, n, func_type="rk4"):
        t = np.zeros(n, dtype=float) 
        states = np.zeros(n * self.size).reshape(self.size, n)

        for i in range(n):
            self.step(h, func_type)
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


    a = LinearEqu(func, 2, [3.0, 0.1])
    output = a.ode(1, 300, "rk4")

    output[0] = output[0] 
    plt.plot(output[0], output[1], label="(Om_0, w_0) = (3, 0.1)")
    plt.xlabel("Theta (rad)")
    plt.ylabel("Omega (rad / s)")
    plt.title("State Space Trajectory Nonlinear Pendulum")
    plt.plot(output[0][0],output[1][0],'ro', label="Starting Point")
    plt.legend()

    #
    #  for i in states:
    #      a.reset(i)
    #      output = a.ode(0.005, 1000, "rk4")
    #      th = output[0]
    #      om = output[1]
    #
    #      th = th % (2 * np.pi)
    #
    #      plt.scatter(th, om, s=2)


    
    #  plt.title("State Space Portrait Nonlinear Damped Pendulum Mod 2pi")
    #  plt.xlabel("Theta (rad)")
    #  plt.ylabel("Omega (rad / sec)")
    #  plt.xlim([0, 2 * np.pi])
    plt.show()









