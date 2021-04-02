import math     # Basic operations - I didn't want to use numpy so its plug and play
import cmath    # Complex sqrt
import copy     # Deep copy for swaps



############################ ODES ####################################################

def err_func_default(y1, y2):
    if type(y1) == int and type(y2) == int:
        return abs(y1 - y2)
    try:
        return math.sqrt(sum([(a - b)**2 for a, b in zip(y1, y2)]))
    except:
        print("Unknown func")

##
# @brief An adaptive time stepping integrator
#
# @param f The ODE function
# @param h The initial time step 
# @param t0 The initial value of t
# @param tf The final value of t
# @param y0 The initial value of y
# @param intfunc The integration function
# @param tollerance Error bound for successive time steps (NOTE: this is not the distance between the actual solution, it is the error between one and two steps)
# @param errfunc The error function (norm / metric)
#
# @return A list of (t, y) values
def adaptive_integrator(f, h, t0, tf, y0, intfunc, tollerance, errfunc):
    if errfunc == None:
        errfunc = err_func_default

    y = y0
    t = t0
    tvals = []
    yvals = []

    while t < tf:

        y1 = intfunc(t, y, h) # One step forward
        y2 = intfunc(t + h/2, intfunc(t, y, h/2), h/2) # Two steps forward
        error = errfunc(y1, y2) # Compute the error
        
        max_recurse = 10
        # Need to decrease step size
        if(error > tollerance):
            while error > tollerance and max_recurse > 0:
                h = h / 2 # Decrease h

                y1 = intfunc(t, y, h)
                y2 = intfunc(t + h/2, intfunc(t, y, h/2), h/2)
                error = errfunc(y1, y2)

                max_recurse -= 1
            if max_recurse <= 0:
                print("Max depth in adaptive stepper exceeded downwards")
            
            if t + h >= tf:
                h = tf - t

            y = intfunc(t, y, h)
            t += h

        # Need to increase step size
        elif(error <= tollerance):

            if t + h >= tf:
                h = tf - t

            # Doing this first because this value is better (even though we need to increase)
            y = intfunc(t, y, h)
            t += h

            while error < tollerance and max_recurse > 0 and t + h <= tf:
                h = h * 2 # Increase h

                y1 = intfunc(t, y, h) 
                y2 = intfunc(t + h/2, intfunc(t, y, h/2), h/2)
                error = errfunc(y1, y2)

                max_recurse -= 1
            if max_recurse <= 0:
                print("Max depth in adaptive stepper exceeded upwards")

        tvals.append(t)
        yvals.append(y)
    return tvals, yvals
    


##
# @brief A Generic integrator that can be used with all the other methds see examples below
#
# @param f y' function (f(y, t))
# @param h Step size
# @param t0 Initial t value
# @param tf Final t value
# @param y0 Initial y value
# @param intfunc The integration function, takes in yn and tn and returns yn+1
#
# @return A list of tuples of (t, y)
def generic_integrator(f, h, t0, tf, y0, intfunc):
    assert h > 0
    y = y0
    t = t0
    tvals = []
    yvals = []
    n = (tf - t) / h
    for i in range(int(n)):
        y = intfunc(t, y, h)
        t += h
        tvals.append(t)
        yvals.append(y)
    return tvals, yvals


## EXAMPLES OF INTEGRATORS
def eulers(f, h0, t0, tf, y0, tollerance = None, errfunc = None):
    intfunc = lambda t, y, h : y + h * f(y, t)
    if tollerance != None:
        return adaptive_integrator(f, h0, t0, tf, y0, intfunc, tollerance, errfunc)
    else:
        return generic_integrator(f, h0, t0, tf, y0, intfunc)

# f is a list of functions - first function is f(y, t) (y'), second is f'(y, t)... 
def taylor_order_n(f: list, h0, t0, tf, y0, tollerance = None, errfunc = None):
    Tn = lambda t, y, h : sum([(h**i)/math.factorial(i + 1) * f[i](y, t) for i in range(len(f))])
    intfunc = lambda t, y, h: y + h * Tn(t, y, h)
    if tollerance != None:
        return adaptive_integrator(f, h0, t0, tf, y0, intfunc, tollerance, errfunc)
    else:
        return generic_integrator(f, h0, t0, tf, y0, intfunc)

def midpoint(f, h0, t0, tf, y0, tollerance = None, errfunc = None):
    intfunc = lambda t, y, h: y + h * f(y + h/2 * f(y, t), t + h/2)
    if tollerance != None:
        return adaptive_integrator(f, h0, t0, tf, y0, intfunc, tollerance, errfunc)
    else:
        return generic_integrator(f, h0, t0, tf, y0, intfunc)

def modified_eulers(f, h0, t0, tf, y0, tollerance = None, errfunc = None):
    intfunc = lambda t, y, h: y + h/2 * (f(y, t) + f(y + h * f(y, h), t + h))
    if tollerance != None:
        return adaptive_integrator(f, h0, t0, tf, y0, intfunc, tollerance, errfunc)
    else:
        return generic_integrator(f, h0, t0, tf, y0, intfunc)

def rk2(f, h0, t0, tf, y0, tollerance = None, errfunc = None):
    intfunc = lambda t, y, h : y + h * f(y + h/2 * f(y, t), t + h/2)
    if tollerance != None:
        return adaptive_integrator(f, h0, t0, tf, y0, intfunc, tollerance, errfunc)
    else:
        return generic_integrator(f, h0, t0, tf, y0, intfunc)

def huens(f, h0, t0, tf, y0, tollerance = None, errfunc = None):
    intfunc = lambda t, y, h : y + h/4 * (f(y, t) + 3 * f(y + 2*h/3 * f(y + h/3 * f(y, t), t + h/3), t + 2*h/3))
    if tollerance != None:
        return adaptive_integrator(f, h0, t0, tf, y0, intfunc, tollerance, errfunc)
    else:
        return generic_integrator(f, h0, t0, tf, y0, intfunc)

def rk4(f, h0, t0, tf, y0, tollerance = None, errfunc = None):
    def intfunc(t, y, h):
        k1 = h * f(y, t)
        k2 = h * f(y + 1/2 * k1, t + h/2)
        k3 = h * f(y + 1/2 * k2, t + h/2)
        k4 = h * f(y + k3, t + h)
        return y + 1/6 * (k1 + 2*k2 + 2*k3 + k4)
    if tollerance != None:
        return adaptive_integrator(f, h0, t0, tf, y0, intfunc, tollerance, errfunc)
    else:
        return generic_integrator(f, h0, t0, tf, y0, intfunc)



############################## Integration ############################################
##
# @brief Numerical Integrator
#
# @param f The function to integrate
# @param (a, b) The domain of the integral
# @param n The number of steps to use
#
# @return Integral from a to b of f
def trapezoid_integration(f, a, b, n):
    h = (b - a) / n
    acc = 0
    for i in range(1, n):
        acc += f(a + i * h)
    return (h / 2) * (f(a) + 2 * acc + f(b))

##
# @brief See trapezoid description
def simpsons_integration(f, a, b, n):
    h = (b - a)/n
    acc1 = 0
    acc2 = 0
    for i in range(1, n):
        if i % 2 == 0:
            acc1 += f(a + i * h)
        else:
            acc2 += f(a + i * h)
    return (h/3) * (f(a) + 2 * acc1 + 4 * acc2 + f(b))

##
# @brief See trapezoid description
def midpoint_integration(f, a, b, n):
    assert n % 2 == 0
    h = (b - a) / (n + 2)
    acc = 0
    for i in range(0, n//2 + 1):
        acc += f(a + (2 * i + 1) * h)
    return 2 * h * acc





################################ Polynomials ###########################################



##
# @brief Defines a natural cubic spline using x and y data points
#
# @param x0 Domain Data
# @param y0 Function data values
#
# @return Coefficient list [[a0, b0, c0, d0], [a1, b1, c1, d1]...] in divided difference for for each interval of x0
def natural_cubic_spline(x0 : list, y0 : list):
    assert len(x0) == len(y0)

    # List of [a b c d] coefficients
    ret = [[y0[i], 0, 0, 0] for i in range(len(x0))]

    h = [x0[i + 1] - x0[i] for i in range(len(x0) - 1)]
    alpha = [3/h[i] * (y0[i + 1] - y0[i]) - 3/h[i - 1] * (y0[i] - y0[i - 1]) for i in range(1, len(x0) - 1)]
    
    l = [0 for i in range(len(x0))]; l[0] = 1
    mu = [0 for i in range(len(x0))]; mu[0] = 0
    z = [0 for i in range(len(x0))]; z[0] = 0

    for i in range(1, len(x0) - 1):
        l[i] = 2 * (x0[i + 1] - x0[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i - 1] - h[i - 1] * z[i - 1]) / l[i]

    l[len(x0) - 1] = 1

    for j in range(len(x0) - 2, -1, -1):
        ret[j][2] = z[j] - mu[j] * ret[j + 1][2]
        ret[j][1] = (ret[j + 1][0]  - ret[j][0])/h[j] - h[j]*(ret[j + 1][2] + 2 * ret[j][2]) / 3
        ret[j][3] = (ret[j + 1][2] - ret[j][2]) / (3 * h[j])

    return ret[0:-1]


##
# @brief Defines a clamped cubic spline using x and y data points as well as derivatives on the boundary
#
# @param x0 Domain data
# @param y0 Function data values
# @param fpx0 F prime at x0[0]
# @param fpxn F prime at x0[-1]
#
# @return coefficients in divided difference form for each interval of x0
def clamped_cubic_spline(x0 : list, y0 : list, fpx0 : float, fpxn : float):

    # List of [a b c d] coefficients
    ret = [[y0[i], 0, 0, 0] for i in range(len(x0))]
    h = [x0[i + 1] - x0[i] for i in range(len(x0) - 1)]

    alpha = [0 for i in range(len(y0))]
    alpha[0] = 3 * (y0[1] - y0[0]) / h[0] - 3 * fpx0
    alpha[-1] = 3 * fpxn - 3 * (y0[-1] - y0[-2])/h[-1]

    for i in range(1, len(x0) - 1):
        alpha[i] = (3 / h[i]) * (y0[i + 1] - y0[i]) - (3 / h[i - 1]) * (y0[i] - y0[i - 1])
    
    l = [0 for i in range(len(x0))]; l[0] = 2 * h[0]
    mu = [0 for i in range(len(x0))]; mu[0] = 0.5
    z = [0 for i in range(len(x0))]; z[0] = alpha[0]/l[0]

    for i in range(1, len(x0) - 1):
        l[i] = 2 * (x0[i + 1] - x0[i - 1]) - h[i - 1] * mu[i -1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

    l[len(x0) - 1] = h[len(x0) - 2] * (2 - mu[len(x0) - 2])
    z[len(x0) - 1] = (alpha[len(x0) - 1] - h[len(x0) - 2] * z[len(x0) - 2]) / l[len(x0) - 1]
    ret[len(x0) - 1][2] = z[len(x0) - 1]

    for i in range(len(x0) - 2, -1, -1):
        ret[i][2] = z[i] - mu[i] * ret[i + 1][2]
        ret[i][1] = (ret[i + 1][0] - ret[i][0]) / h[i] - h[i] * (ret[i + 1][2] + 2 * ret[i][2]) / 3
        ret[i][3] = (ret[i + 1][2] - ret[i][2]) / (3 * h[i])

    return ret[0:-1]




##
# @brief Hermite polynomial coefficients and evaluation
#
# @param x0 Domain Data values
# @param y0 Function Data Values
# @param yprime0 Derivative Data Values
# @param x Optional value of x to evaluate
#
# @return If no x given, return coefficients in divided difference form and new array of z = [x0, x0, x1, x1 ...]
# If x is given, evaluates the hermite polynomial at x and returns coefficients, eval, z
def hermite(x0 : list, y0 : list, yprime0 : list, x = None):
    z = [0 for i in range(2 * len(x0))]
    Q = [[0 for j in range(2 * (i + 1))] for i in range(2 * len(y0))]
    for i in range(len(x0)):
        z[2 * i] = x0[i]
        z[2 * i + 1] = x0[i]
        Q[2 * i][0] = y0[i]
        Q[2 * i + 1][0] = y0[i]
        Q[2 * i + 1][1] = yprime0[i]
        
        if i != 0:
            Q[2 * i][1] = (Q[2 * i][0] - Q[2 * i - 1][0])/(z[2 * i] - z[2 * i - 1])
    for i in range(2, 2 * len(x0)):
        for j in range(2, i + 1):
            Q[i][j] = (Q[i][j - 1] - Q[i - 1][j - 1]) / (z[i] - z[i - j])

    coefs = [Q[i][i] for i in range(len(Q))]
    if x == None:
        return coefs, z
    return coefs, divdiff_hermite_eval(coefs, x, z), z
        

##
# @brief Divided Difference Method
#
# @param x0 a list of x0 values
# @param y0 a list of y0 values
# @param x The value of x to calculate
#
# @return DD coefficients (in the form a0 + a1(x - x0) + a2(x - x0)(x - x1) ..., and the function evaluated at x
def divdiff_newton(x0: list, y0: list, x):
    assert len(x0) == len(y0)

    # Initializes an array like such:
    # [
    #   [y0]
    #   [y1 0]
    #   [y2 0 0]
    #   ...
    #   [yn 0 0 0 ... 0]
    # ]
    # F[0] := [y0]
    # F[1] := [y1, 0]
    F = [[(y0[i] if j == 0 else 0) for j in range(i + 1)] for i in range(len(y0))]
    
    for i in range(1, len(x0)):
        for j in range(1, i + 1):
            F[i][j] = (F[i][j-1] - F[i-1][j-1])/(x0[i]-x0[i-j])

    coefs = [F[i][i] for i in range(len(F))]
    
    # If no x to evaluate specified, just return coefficients
    if x == None:
        return coefs
    
    return coefs, divdiff_eval(coefs, x, x0)

##
# @brief Evaluates a polynomial in the form of divided difference
#
# @param coefs - Div diff coeffients (a0, a1, a2...) such that a0 + a1(x - x0) + a2(x - x0)(x - x1)...
# @param x The value of x to eval
#
# @return Evaluated x
def divdiff_eval(coefs, x, x0):
    p = 0
    for i in range(len(x0)):
        p_ = coefs[i]
        for j in range(i):
            p_ *= (x - x0[j])
        p += p_
    return p

##
# @brief This is the same thing as divdiff eval - just caled it hermite
#
# @param coefs Div Diff coefficients 
# @param x The value of x to eval
# @param z0 List of x0 values
#
# @return Evaluated x
def divdiff_hermite_eval(coefs, x, z0):
    p = 0
    for i in range(len(z0)):
        p_ = coefs[i]
        for j in range(i):
            p_ *= (x - z0[j])
        p += p_
    return p


##
# @brief Nevilles method evaluating x
#
# @param x  The value to evaluate
# @param x0 Data in
# @param y0 Data in
#
# @return table
def nevilles(x, x0: list, y0: list): 
    
    assert len(x0) == len(y0)
    Q = [[(y0[i] if j == 0 else 0) for j in range(i + 1)] for i in range(len(y0))]
    print(Q)

    # The body - pretty long formula, wanted to condense to a function
    f = lambda i, j, Q : ((x-x0[i-j]) * Q[i][j-1] - (x-x0[i]) * Q[i-1][j-1]) / (x0[i]-x0[i-j])

    for i in range(1, len(x0)):
        for j in range(1, i + 1):
            Q[i][j] = f(i, j, Q)

    return Q


##
# @brief A Horner Polynomial is a "smart" polynomial
# It evaluates itself at x but if x is passed again,
# it doesn't go through the trouble of evaluating itself again
# so if a is a Horner_polynomial,
# a.eval_f(3)
# a.eval_fprime(3)
# only executes horner's method once
class Horner_Polynomial:
    def __init__(self, coefs):
        self.coefs = coefs
        self.y = None
        self.yp = None
        self.last_x = None

    ##
    # @brief Gets f
    # @param x The value to evaluate
    # @return f(x)
    def eval_f(self, x):
        self.__eval(x)
        return self.y

    ##
    # @brief Get's f prime
    # @param x The value to evaluate
    # @return f'(x)
    def eval_fprime(self, x):
        self.__eval(x)
        return self.yp


    ##
    # @brief Evaluate self at x if x is unique 
    # @param x The value to evaluate at
    def __eval(self, x):
        if x == self.last_x:
            return 
        self.last_x = x
        self.y, self.yp = horners(len(self.coefs) - 1, self.coefs, x)

    ##
    # @brief Compute the roots using horners and newtons 
    #
    # @param list A list of initial starting points. Note that the polynomial is
    # reduced every step, so it is not a problem to repeat values for x
    # @param e Tollerance
    # @param n Iterations
    #
    # @return [list of roots to associated starting x, reduced polynomial coefficients]
    def roots_newton(self, x: list, e = 0, n = float('inf')):
        ret = []
        coef = copy.deepcopy(self.coefs)
        for i in x:
            x0, err = newtons_method(self.eval_f, self.eval_fprime, i, e, n)
            ret.append(x0)
            _, _, self.coefs = horners(len(self.coefs) - 1, self.coefs, x0, True)
        coef, self.coefs = self.coefs, coef
        return ret, coef

    ##
    # @brief Compute the roots using horners and mullers
    #
    # @param list A list of initial three vectors of initial points
    # Ex: [(1, 2, 3), (2, 3, 4), ...]
    # @param e Tolerance
    # @param n Iterations
    #
    # @return [list of roots to associated starting values, reduced polynomial coefficients]
    def roots_muller(self, x: list, e = 0, n = float('inf')):
        ret = []
        coef = copy.deepcopy(self.coefs)
        for i in x:
            x0, err = mullers(self, i[0], i[1], i[2], e, n)
            ret.append(x0)
            _, _, self.coefs = horners(len(self.coefs) - 1, self.coefs, x0, True)
        coef, self.coefs = self.coefs, coef
        return ret, coef

##
# @brief Evaluates a polynomial and its derivative at x0
#
# @param int Degree
# @param list coefficients (in assending power order)
# @param float value to eval
# @param save_Q If True, returns a reduced polynomial (after dividing by (x - x0))
#
# @return f(x0), f'(x0), (optional) Reduced Coefficients
def horners(n: int, coef: list, x0: float, save_Q = False):
    assert n == len(coef) - 1

    if n == 0: return coef[n], 0.0

    if save_Q:
        Q = []

    y = coef[n]
    z = coef[n]
    if save_Q: Q.append(y)
    for j in range(n - 1, 0, -1):
        y = x0 * y + coef[j]
        z = x0 * z + y
        if save_Q: Q.insert(0, y)

    y = x0 * y + coef[0]

    if save_Q:
        return y, z, Q
    return y, z


##
# @brief Evaluates a polynomial like horners but with complex values
#
# @param a Horner Polynomial class
# @param p0 starting point0 
# @param p1 starting point1
# @param p2 starting point2
# @param tol Tollerance
# @param n number of itterations
#
# @return root, error
def mullers(a : Horner_Polynomial, p0, p1, p2, tol = 0, n = float('inf')):
    assert p0 != p1
    assert p1 != p2
    assert p2 != p0
    assert tol > 0 or n < float('inf')
    h1 = p1 - p0
    h2 = p2 - p1
    f = lambda x : a.eval_f(x)
    del1 = (f(p1) - f(p0)) / h1
    del2 = (f(p2) - f(p1)) / h2
    d = (del2 - del1) / (h2 + h1)
    i = 3

    p = None
    h = None

    while i < n:
        b = del2 + h2 * d
        temp = b * b - 4 * f(p2) * d
        D = cmath.sqrt(temp)
        if abs(b - D) <= abs(b + D):
            E = b + D
        else:
            E = b - D
        h = -2 * f(p2) / E
        p = p2 + h

        if abs(h) < tol:
            return p, abs(h)

        p0 = p1
        p1 = p2
        p2 = p
        h1 = p1 - p0
        h2 = p2 - p1
        del1 = (f(p1) - f(p0)) / h1
        del2 = (f(p2) - f(p1)) / h2
        d = (del2 - del1)/(h2 + h1)
        i = i + 1
    return p, abs(h)





# TODO Turn these all into itterative instead of recursive functions

################################# Single Variable Solutions #############################
"""
All of these problems will not terminate because
tolerance is 0 and n = infinity, so you MUST set at
least one
"""

def bisection_method(f, a0, b0, tol = 0, n = float('inf'), verbose = False):
    middle = (a0 + b0) / 2              
    fa, fb, fm = f(a0), f(b0), f(middle)
    
    assert b0 > a0                          
    assert fa * fb < 0                      

    # Termination
    if n <= 1 or abs(fm) < tol: 
        return middle, abs(fm)

    # Recursive Call
    a0, b0 = (middle, b0) if fb * fm < 0 else (a0, middle)
    return bisection_method(f, a0, b0, tol, n - 1) 

def fixed_point_method(f, a0, tol = 0, n = float('inf'), verbose = False):
    try:
        fa = f(a0)
    except OverflowError: 
        print("Sequence does not converge")
        return float('inf'), float('inf')

    # Termination
    if n <= 1 or abs(fa - a0) < tol:
        return fa, abs(fa - a0)

    # Recursive Call
    return fixed_point_method(f, fa, tol, n - 1, verbose)


def newtons_method(f, f_prime, p0, tol = 0, n = float('inf'), verbose = False):
    try:
        fp = f(p0)
        fpp = f_prime(p0)
        p = p0 - fp / fpp
    except OverflowError:
        print("Sequence does not converge")
        return float('inf'), float('inf')
    except ZeroDivisionError:
        print("Did not satisfy second condition for newton's convergence")
        return None, None
    
    # Termination
    if n <= 1 or abs(p - p0) < tol: 
        return p, abs(p - p0)

    # Recursive Call
    return newtons_method(f, f_prime, p, tol, n - 1)

def newtons_2_13_method(f, f_prime, f_double_prime, p0, tol = 0, n = float('inf'), verbose = False):
    try:
        f_ = f(p0)
        fp = f_prime(p0)
        fpp = f_double_prime(p0)
        p = p0 - (f_ * fp) / ((fp)**2 - f_ * fpp)
    except OverflowError:
        print("Sequence does not converge")
        return float('inf'), float('inf')
    except ZeroDivisionError:
        print("Did not satisfy second condition for newton's convergence")
        return None, None
    
    # Termination
    if n <= 1 or abs(p - p0) < tol: 
        return p, abs(p - p0)

    # Recursive Call
    return newtons_2_13_method(f, f_prime, f_double_prime, p, tol, n - 1)

def secant_method(f, p0, p1, tol = 0, n = float('inf'), verbose = False):

    def __sec_recurs(p0_, p1_, q0_, q1_, n_):
        try:
            p = p1_ - q1_ * (p1_ - p0_) / (q1_ - q0_)
        except OverflowError:
            print("Sequence does not converge")
            return float('inf'), float('inf')

        # Termination
        if n_ <= 2 or abs(p - p1_) < tol:
            return p, abs(p - p1_)

        # Recursive Call
        return __sec_recurs(p1_, p, q1_, f(p), n_ - 1)

    return __sec_recurs(p0, p1, f(p0), f(p1), n)


def false_position_method(f, p0, p1, tol = 0, n = float('inf'), verbose = False):

    def __false_p_recurs(p0_, p1_, q0_, q1_, n_):
        try:
            p = p1_ - q1_ * (p1_ - p0_)/(q1_ - q0_)
            q = f(p)
        except OverflowError:
            print("Sequence does not converge")
            return float('inf'), float('inf')

        # Termination
        if n <= 2 or abs(p - p1_) < tol:
            return p, abs(p - p1_)

        # Recursive Call
        if q * q1_ < 0: p0_, q0_ = p1_, q1_
        return __false_p_recurs(p0, p, q0, q, n - 1)

    return __false_p_recurs(p0, p1, f(p0), f(p1), n)

def steffensens_method(f, p0, tol = 0, n = float('inf'), verbose = False):
    p1 = f(p0)
    p2 = f(p1)
    p = p0 - (p1 - p0)**2 / (p2 - 2 * p1 + p0)

    if n <= 1 or abs(p - p0) < tol:
        return p, abs(p - p0)

    return steffensens_method(f, p, tol, n - 1)

def horners_method(degree, coef, x0):
    assert type(degree) == int
    assert len(coef) - 1 <= degree

    def __recurs(y, z, i = len(coef) - 2):
        if i <= 0:
            return x0 * y + coef[-1], z
        return __recurs(x0 * y + coef[j], x0 * z + y, i - 1)

    return __recurs(coef[0], coef[0])



############################################### UTILITY ################################
# A Few Clean Printing methods
def problem(name):
    print("\n===============================")
    print("Problem: ", name)

# Prints a polynomial as coefficients
def poly_print(coef):
    ret = ""
    n = len(coef) - 1
    for i in range(n):
        ret += f"{coef[n - i]}x^{n - i} + "
    ret += f"{coef[0]}"
    return ret

# Prints polynomial in the other form (a0 + a1(x - x0) + a2(x - x0)(x - x1)...)
def div_poly_print(coef, x0):
    p = ""
    n = len(coef) - 1
    for i in range(len(coef)):
        p += f"{coef[i]:.3f}"
        for j in range(i):
            p += f"(x - {x0[j]:.3f})"
        if i != n:
            p += " + "
    return p

# Prints a factored polynomial
def poly_print_factored(zeros):
    ret = "C" # Technically we add a constant term in the front - although I never really need this
    for i in zeros:
        ret += f"(x-{i:.3f})" # assuming conjigate exists, so x - a + b and x - a - b are swapped
    return ret
