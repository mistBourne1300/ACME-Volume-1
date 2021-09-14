# matplotlib_intro.py
"""Python Essentials: Intro to Matplotlib.
<Name>
<Class>
<Date>
"""
import numpy as np
from matplotlib import pyplot as plt
from numpy.core.fromnumeric import mean, var

# Problem 1
def var_of_means(n):
    """Construct a random matrix A with values drawn from the standard normal
    distribution. Calculate the mean value of each row, then calculate the
    variance of these means. Return the variance.

    Parameters:
        n (int): The number of rows and columns in the matrix A.

    Returns:
        (float) The variance of the means of each row.
    """
    A = np.random.normal(size = (n,n))
    rows = np.mean(A, axis = 1)
    #print(rows)
    return np.var(rows)

def prob1():
    """Create an array of the results of var_of_means() with inputs
    n = 100, 200, ..., 1000. Plot and show the resulting array.
    """
    nums = [i for i in range(100,1001,100)]
    means_array = [var_of_means(n) for n in nums]
    plt.plot(nums, means_array)
    plt.show()



# Problem 2
def prob2():
    """Plot the functions sin(x), cos(x), and arctan(x) on the domain
    [-2pi, 2pi]. Make sure the domain is refined enough to produce a figure
    with good resolution.
    """
    domain = np.linspace(-1*np.pi, np.pi, 1000)
    sine = np.sin(domain)
    cosine = np.cos(domain)
    arctangent = np.arctan(domain)
    plt.plot(domain, sine)
    plt.plot(domain, cosine)
    plt.plot(domain, arctangent)
    plt.show()


# Problem 3
def prob3():
    """Plot the curve f(x) = 1/(x-1) on the domain [-2,6].
        1. Split the domain so that the curve looks discontinuous.
        2. Plot both curves with a thick, dashed magenta line.
        3. Set the range of the x-axis to [-2,6] and the range of the
           y-axis to [-6,6].
    """
    domain1 = np.linspace(-2,1,500)
    domain2 = np.linspace(1,6,500)
    plt.plot(domain1, 1/(domain1 - 1), "m--", linewidth = 4)
    plt.plot(domain2, 1/(domain2 - 1), "m--", linewidth = 4)
    plt.xlim(-2,6)
    plt.ylim(-6,6)
    plt.show()



# Problem 4
def prob4():
    """Plot the functions sin(x), sin(2x), 2sin(x), and 2sin(2x) on the
    domain [0, 2pi].
        1. Arrange the plots in a square grid of four subplots.
        2. Set the limits of each subplot to [0, 2pi]x[-2, 2].
        3. Give each subplot an appropriate title.
        4. Give the overall figure a title.
        5. Use the following line colors and styles.
              sin(x): green solid line.
             sin(2x): red dashed line.
             2sin(x): blue dashed line.
            2sin(2x): magenta dotted line.
    """
    domain = np.linspace(0,2*np.pi)
    plt.subplot(221)
    plt.plot(domain, np.sin(domain), 'g-')
    plt.axis([0,2*np.pi,-2,2])
    
    plt.subplot(222)
    plt.plot(domain, np.sin(2*domain), 'r--')
    plt.axis([0,2*np.pi,-2,2])
    
    plt.subplot(223)
    plt.plot(domain, 2*np.sin(domain), 'b--')
    plt.axis([0,2*np.pi,-2,2])
    
    plt.subplot(224)
    plt.plot(domain, 2*np.sin(2*domain), 'm:')
    plt.axis([0,2*np.pi,-2,2])

    plt.show()




# Problem 5
def prob5():
    """Visualize the data in FARS.npy. Use np.load() to load the data, then
    create a single figure with two subplots:
        1. A scatter plot of longitudes against latitudes. Because of the
            large number of data points, use black pixel markers (use "k,"
            as the third argument to plt.plot()). Label both axes.
        2. A histogram of the hours of the day, with one bin per hour.
            Label and set the limits of the x-axis.
    """
    FARS = np.load("FARS.npy")
    #print(FARS)
    long = FARS[:,1]
    #print(long)
    lat = FARS[:,2]
    hours = FARS[:,0]
    plt.subplot(121)
    plt.plot(long, lat, 'k,')
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.axis("equal")

    plt.subplot(122)
    plt.hist(hours, bins = 24, range = [0,23])
    plt.show()


# Problem 6
def prob6():
    """Plot the function f(x,y) = sin(x)sin(y)/xy on the domain
    [-2pi, 2pi]x[-2pi, 2pi].
        1. Create 2 subplots: one with a heat map of f, and one with a contour
            map of f. Choose an appropriate number of level curves, or specify
            the curves yourself.
        2. Set the limits of each subplot to [-2pi, 2pi]x[-2pi, 2pi].
        3. Choose a non-default color scheme.
        4. Add a colorbar to each subplot.
    """
    x = np.linspace(-2*np.pi, 2*np.pi, 100)
    y = x.copy()
    X, Y = np.meshgrid(x,y)
    plt.subplot(121)
    Z = (np.sin(X) * np.sin(Y))/(X*Y)
    plt.pcolormesh(X, Y, Z)

    plt.subplot(122)
    plt.contour(X,Y,Z)

    plt.show()

if __name__ == "__main__":
    prob1()
    prob2()
    prob3()
    prob4()
    prob5()
    prob6()

