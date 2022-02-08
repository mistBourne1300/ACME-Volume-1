# montecarlo_integration.py
"""Volume 1: Monte Carlo Integration.
<Name>
<Class>
<Date>
"""
from multiprocessing.sharedctypes import Value
import numpy as np
from scipy import linalg as la
from scipy import stats
import matplotlib.pyplot as plt

# Problem 1
def ball_volume(n, N=int(1e4)):
    """Estimate the volume of the n-dimensional unit ball.

    Parameters:
        n (int): The dimension of the ball. n=2 corresponds to the unit circle,
            n=3 corresponds to the unit sphere, and so on.
        N (int): The number of random points to sample.

    Returns:
        (float): An estimate for the volume of the n-dimensional unit ball.
    """
    # get the n-dimensional points
    points = np.random.uniform(-1,1,(n,N))

    # get the lengths
    lengths = la.norm(points, axis = 0)

    # count how many of them are inside the unit ball
    within = np.count_nonzero(lengths < 1)

    # return the estimated volume
    return 2**n * (within/N)



# Problem 2
def mc_integrate1d(f, a, b, N=10000):
    """Approximate the integral of f on the interval [a,b].

    Parameters:
        f (function): the function to integrate. Accepts and returns scalars.
        a (float): the lower bound of interval of integration.
        b (float): the lower bound of interval of integration.
        N (int): The number of random points to sample.

    Returns:
        (float): An approximation of the integral of f over [a,b].

    Example:
        >>> f = lambda x: x**2
        >>> mc_integrate1d(f, -4, 2)    # Integrate from -4 to 2.
        23.734810301138324              # The true value is 24.
    """

    # get points
    points = np.random.uniform(a,b,N)

    # return extimated volume
    return (b-a)*np.sum([f(x) for x in points])/N



# Problem 3
def mc_integrate(f, mins, maxes, N=10000):
    """Approximate the integral of f over the box defined by mins and maxs.

    Parameters:
        f (function): The function to integrate. Accepts and returns
            1-D NumPy arrays of length n.
        mins (list): the lower bounds of integration.
        maxs (list): the upper bounds of integration.
        N (int): The number of random points to sample.

    Returns:
        (float): An approximation of the integral of f over the domain.

    Example:
        # Define f(x,y) = 3x - 4y + y^2. Inputs are grouped into an array.
        >>> f = lambda x: 3*x[0] - 4*x[1] + x[1]**2

        # Integrate over the box [1,3]x[-2,1].
        >>> mc_integrate(f, [1, -2], [3, 1])
        53.562651072181225              # The true value is 54.
    """
    # if the mins and maxes don't have the same dimension, we have a problem
    if len(maxes) != len(mins):
        raise ValueError("mins and maxes must have same dimension")
    
    # get the dimension so we don't have to use len() all the time
    n = len(maxes)
    
    #initialize a volume of 1 and an empty points array
    volume = 1
    points = []

    # calculate volume and append uniform points to the points array
    for i in range(n):
        volume *= maxes[i] - mins[i]
        points.append(np.random.uniform(mins[i], maxes[i], N))
    
    # make points an array so we can take its transpose
    points = np.array(points)

    # use 6.1 to estimate the integral
    return volume * np.sum([f(x) for x in points.T])/N



# Problem 4
def prob4():
    """Let n=4 and Omega = [-3/2,3/4]x[0,1]x[0,1/2]x[0,1].
    - Define the joint distribution f of n standard normal random variables.
    - Use SciPy to integrate f over Omega.
    - Get 20 integer values of N that are roughly logarithmically spaced from
        10**1 to 10**5. For each value of N, use mc_integrate() to compute
        estimates of the integral of f over Omega with N samples. Compute the
        relative error of estimate.
    - Plot the relative error against the sample size N on a log-log scale.
        Also plot the line 1 / sqrt(N) for comparison.
    """
    # get dimension and define f
    n = 4
    f = lambda x: 1/((2*np.pi)**(len(x)/2)) * np.exp(-np.dot(x,x)/2)

    # mins and maxes of the integral
    mins = np.array([-3/2, 0, 0, 0])
    maxes = np.array([3/4, 1, 1/2, 1])

    # get the "true" value of 
    means, cov = np.zeros(n), np.eye(n)
    treu = stats.mvn.mvnun(mins, maxes, means, cov)[0]
    eNNs = np.logspace(1,5, 20, base = 10, dtype = int)
    errs = []

    # calculate the relative errors
    for N in eNNs:
        _F_ = mc_integrate(f, mins, maxes, N = N)
        print(_F_)
        errs.append(np.abs(treu - _F_)/np.abs(treu))

    # plot the relative errors
    plt.loglog(eNNs, errs)
    plt.loglog(eNNs, 1/np.sqrt(eNNs))
    plt.title("Relative Error in Monte Carlo Integration")
    plt.legend(['Relative Error', '1/sqrt(N)'])
    plt.show()
    




if __name__ == "__main__":
    # d = 4
    # print(f'd = {d}: {ball_volume(d, int(1e7))}')

    # f1 = lambda x: x**2
    # f2 = lambda x: np.sin(x)
    # f3 = lambda x: 1/x
    # print(f'f1: {mc_integrate1d(f1,-4,2)}')
    # print(f'f2: {mc_integrate1d(f2,-2*np.pi,2*np.pi)}')
    # print(f'f3: {mc_integrate1d(f3,1,10)}')

    # f4 = lambda x: x[0]**2 + x[1]**2
    # f5 = lambda x: 3*x[0] + 4*x[1] + x[1]**2
    # f6 = lambda x: x[0] + x[1] - x[3]*(x[2]**2)

    # print(mc_integrate(f4, [0,0], [1,1]))

    prob4()