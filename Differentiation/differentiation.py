# differentiation.py
"""Volume 1: Differentiation.
<Name>
<Class>
<Date>
"""
import sympy as sy
import numpy as np
import matplotlib.pyplot as plt
import os
from autograd import elementwise_grad as ewok
from autograd import numpy as anp
from autograd import grad
from time import time

# Problem 1
def prob1():
    """Return the derivative of (sin(x) + 1)^sin(cos(x)) using SymPy."""
    x = sy.symbols('x')
    expr = (sy.sin(x) + 1)**(sy.sin(sy.cos(x)))
    deriv = sy.diff(expr, x)
    f = sy.lambdify(x, expr, 'numpy')
    f_prime = sy.lambdify(x, deriv, 'numpy')
    # domain = np.linspace(-np.pi, np.pi, 1000)
    # plt.plot(domain, f(domain), 'k')
    # plt.plot(domain, f_prime(domain), 'r')
    # plt.title("(sin(x) + 1)^sin(cos(x)) and its derivative")
    # plt.legend(['f', 'f_prime'])
    # plt.show()
    return f_prime


# Problem 2
def fdq1(f, x, h=1e-5):
    """Calculate the first order forward difference quotient of f at x."""
    return (f(x+h) - f(x))/h


def fdq2(f, x, h=1e-5):
    """Calculate the second order forward difference quotient of f at x."""
    return (-3*f(x) + 4*f(x + h) - f(x + 2*h))/(2*h)


def bdq1(f, x, h=1e-5):
    """Calculate the first order backward difference quotient of f at x."""
    return (f(x) - f(x-h))/h


def bdq2(f, x, h=1e-5):
    """Calculate the second order backward difference quotient of f at x."""
    return (3*f(x) - 4*f(x - h) + f(x - 2*h))/(2*h)

def cdq2(f, x, h=1e-5):
    """Calculate the second order centered difference quotient of f at x."""
    return (f(x+h) - f(x-h))/(2*h)


def cdq4(f, x, h=1e-5):
    """Calculate the fourth order centered difference quotient of f at x."""
    return (f(x - 2*h) - 8*f(x - h) + 8*f(x + h) - f(x + 2*h))/(12*h)


# Problem 3
def prob3(x0):
    """Let f(x) = (sin(x) + 1)^(sin(cos(x))). Use prob1() to calculate the
    exact value of f'(x0). Then use fdq1(), fdq2(), bdq1(), bdq2(), cdq1(),
    and cdq2() to approximate f'(x0) for h=10^-8, 10^-7, ..., 10^-1, 1.
    Track the absolute error for each trial, then plot the absolute error
    against h on a log-log scale.

    Parameters:
        x0 (float): The point where the derivative is being approximated.
    """
    f = lambda x: (np.sin(x) + 1)**(np.sin(np.cos(x)))
    haghtches = np.logspace(-8, 0, 9)
    f_prime_x0 = prob1()(x0)
    fdq1e, fdq2e, bdq1e, bdq2e, cdq2e, cdq4e = [], [], [], [], [], []
    for h in haghtches:
        fdq1e.append(np.abs(fdq1(f, x0, h) - f_prime_x0))
        fdq2e.append(np.abs(fdq2(f, x0, h) - f_prime_x0))
        bdq1e.append(np.abs(bdq1(f, x0, h) - f_prime_x0))
        bdq2e.append(np.abs(bdq2(f, x0, h) - f_prime_x0))
        cdq2e.append(np.abs(cdq2(f, x0, h) - f_prime_x0))
        cdq4e.append(np.abs(cdq4(f, x0, h) - f_prime_x0))
    
    plt.loglog(haghtches, fdq1e)
    plt.loglog(haghtches, fdq2e)
    plt.loglog(haghtches, bdq1e)
    plt.loglog(haghtches, bdq2e)
    plt.loglog(haghtches, cdq2e)
    plt.loglog(haghtches, cdq4e)
    plt.legend(['fdq1e', 'fdq2e', 'bdq1e', 'bdq2e', 'cdq2e', 'cdq4e'])
    plt.xlabel('h-value')
    plt.ylabel('absolute error')
    plt.title('error v h-value')
    plt.show()


# Problem 4
def prob4():
    """The radar stations A and B, separated by the distance 500m, track a
    plane C by recording the angles alpha and beta at one-second intervals.
    Your goal, back at air traffic control, is to determine the speed of the
    plane.

    Successive readings for alpha and beta at integer times t=7,8,...,14
    are stored in the file plane.npy. Each row in the array represents a
    different reading; the columns are the observation time t, the angle
    alpha (in degrees), and the angle beta (also in degrees), in that order.
    The Cartesian coordinates of the plane can be calculated from the angles
    alpha and beta as follows.

    x(alpha, beta) = a tan(beta) / (tan(beta) - tan(alpha))
    y(alpha, beta) = (a tan(beta) tan(alpha)) / (tan(beta) - tan(alpha))

    Load the data, convert alpha and beta to radians, then compute the
    coordinates x(t) and y(t) at each given t. Approximate x'(t) and y'(t)
    using a first order forward difference quotient for t=7, a first order
    backward difference quotient for t=14, and a second order centered
    difference quotient for t=8,9,...,13. Return the values of the speed at
    each t.
    """
    a = 500
    x_pos = lambda alpha, beta: a * np.tan(beta) / (np.tan(beta) - np.tan(alpha))
    y_pos = lambda alpha, beta: a * np.tan(beta)*np.tan(alpha) / (np.tan(beta) - np.tan(alpha))
    dat = np.load("plane.npy")
    # print(dat)
    alphas, betas = np.deg2rad(dat[:,1]), np.deg2rad(dat[:,2])
    # print(alphas, betas)
    # print("\n\n")
    x_posse = x_pos(alphas, betas)
    y_posse = y_pos(alphas, betas)
    # print(x_posse)
    # print(y_posse)
    x_primes = [x_posse[1] - x_posse[0]]
    y_primes = [y_posse[1] - y_posse[0]]
    for i in range(1,len(x_posse)-1):
        x_primes.append((x_posse[i+1] - x_posse[i-1])/2)
        y_primes.append((y_posse[i+1] - y_posse[i-1])/2)
    x_primes.append(x_posse[-1] - x_posse[-2])
    y_primes.append(y_posse[-1] - y_posse[-2])
    x_primes = np.array(x_primes)
    y_primes = np.array(y_primes)
    # print(x_primes)
    # print(y_primes)
    # print()
    return np.sqrt(x_primes**2 + y_primes**2)



# Problem 5
def jacobian_cdq2(f, x, h=1e-5):
    """Approximate the Jacobian matrix of f:R^n->R^m at x using the second
    order centered difference quotient.

    Parameters:
        f (function): the multidimensional function to differentiate.
            Accepts a NumPy (n,) ndarray and returns an (m,) ndarray.
            For example, f(x,y) = [x+y, xy**2] could be implemented as follows.
            >>> f = lambda x: np.array([x[0] + x[1], x[0] * x[1]**2])
        x ((n,) ndarray): the point in R^n at which to compute the Jacobian.
        h (float): the step size in the finite difference quotient.

    Returns:
        ((m,n) ndarray) the Jacobian matrix of f at x.
    """
    I = np.identity(len(x))
    Jacob = []
    for i in range(len(f(x))):
        ithrow = []
        for ej in I:
            ithrow.append((f(x+h*ej)[i] - f(x-h*ej)[i])/(2*h))
        
        Jacob.append(ithrow)
    return np.array(Jacob)


# Problem 6
def cheb_poly(x, n):
    """Compute the nth Chebyshev polynomial at x.

    Parameters:
        x (autograd.ndarray): the points to evaluate T_n(x) at.
        n (int): The degree of the polynomial.
    """
    if n == 0:
        return anp.ones_like(x)
    if n == 1:
        return x
    return 2*x*cheb_poly(x,n-1) - cheb_poly(x, n-2)


def prob6():
    """Use Autograd and cheb_poly() to create a function for the derivative
    of the Chebyshev polynomials, and use that function to plot the derivatives
    over the domain [-1,1] for n=0,1,2,3,4.
    """
    domain = anp.array(np.linspace(-1,1))
    for i in range(5):
        plt.plot(domain, cheb_poly(domain,i))
        deriv = ewok(cheb_poly)
        plt.plot(domain, deriv(domain, i))
    plt.show()



# Problem 7
def prob7(N=200):
    """Let f(x) = (sin(x) + 1)^sin(cos(x)). Perform the following experiment N
    times:

        1. Choose a random value x0.
        2. Use prob1() to calculate the “exact” value of f′(x0). Time how long
            the entire process takes, including calling prob1() (each
            iteration).
        3. Time how long it takes to get an approximation of f'(x0) using
            cdq4(). Record the absolute error of the approximation.
        4. Time how long it takes to get an approximation of f'(x0) using
            Autograd (calling grad() every time). Record the absolute error of
            the approximation.

    Plot the computation times versus the absolute errors on a log-log plot
    with different colors for SymPy, the difference quotient, and Autograd.
    For SymPy, assume an absolute error of 1e-18.
    """
    f = lambda x: (anp.sin(x) + 1)**(anp.sin(anp.cos(x)))
    sympy_times = []
    sympy_errs = [1e-18 for i in range(N)]
    cdq4_times = []
    cdq4_errs = []
    auto_times = []
    auto_errs = []
    for i in range(N):
        x0 = 2*np.pi*(np.random.rand()-.5)
        start = time()
        f_prime_x0 = prob1()(x0)
        sympy_times.append(time()-start)


        start = time()
        cdq4_errs.append(np.abs(f_prime_x0 - cdq4(f,x0)))
        cdq4_times.append(time() - start)
        

        start = time()
        auto_errs.append(np.abs(f_prime_x0 - grad(f)(x0)))
        auto_times.append(time() - start)
    
    plt.xlabel("Computation Time")
    plt.ylabel("Absolute Error")
    plt.scatter(sympy_times, sympy_errs, c = 'b', s = 10)
    plt.scatter(cdq4_times, cdq4_errs, c = 'r', s = 10)
    plt.scatter(auto_times, auto_errs, c = 'k', s = 10)
    plt.legend(['sympy', 'cdq4', 'autograd'])
    plt.xscale('log')
    plt.yscale('log')
    plt.show()


if __name__ == "__main__":
    # os.chdir("/Users/chase/Desktop/Math345Volume1/byu_vol1/Differentiation")
    # prob1()

    # domain = np.linspace(-np.pi, np.pi, 1000)
    # f = lambda x: (np.sin(x) + 1)**(np.sin(np.cos(x)))
    # plt.plot(domain, f(domain), 'k')
    # plt.plot(domain, fdq1(f,domain))
    # plt.plot(domain, fdq2(f, domain))
    # plt.plot(domain, bdq1(f, domain))
    # plt.plot(domain, bdq2(f, domain))
    # plt.plot(domain, cdq2(f, domain))
    # plt.plot(domain, cdq4(f, domain))
    # plt.legend(['f', 'fdq1', 'fdq2', 'bdq1', 'bdq2', 'cdq2', 'cdq4'])
    # plt.show()

    # prob3(1)

    # print(prob4())

    # f = lambda x: np.array([x[0] + x[1], x[0] * x[1]**2], x[2])
    # print(jacobian_cdq2(f,np.array([1,1, 0])))

    # prob6()
    # prob7()
    pass