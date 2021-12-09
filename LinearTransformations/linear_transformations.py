# linear_transformations.py
"""Volume 1: Linear Transformations.
<Name>
<Class>
<Date>
"""

from random import random
import numpy as np
from matplotlib import pyplot as plt
import time


# Problem 1
def stretch(A, a, b):
    """Scale the points in A by a in the x direction and b in the
    y direction.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): scaling factor in the x direction.
        b (float): scaling factor in the y direction.
    """
    STRETCH = np.array([[a,0],[0,b]])
    return STRETCH @ A


def shear(A, a, b):
    """Slant the points in A by a in the x direction and b in the
    y direction.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): scaling factor in the x direction.
        b (float): scaling factor in the y direction.
    """
    return np.array([[1,a],[b,1]]) @ A

def reflect(A, a, b):
    """Reflect the points in A about the line that passes through the origin
    and the point (a,b).

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): x-coordinate of a point on the reflecting line.
        b (float): y-coordinate of the same point on the reflecting line.
    """
    return ((1/a**2 + b**2) * np.array([[a**2 - b**2, 2*a*b],[2*a*b, b**2 - a**2]])) @ A

def rotate(A, theta):
    """Rotate the points in A about the origin by theta radians.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        theta (float): The rotation angle in radians.
    """
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) @ A


# Problem 2
def solar_system(T, x_e, x_m, omega_e, omega_m):
    """Plot the trajectories of the earth and moon over the time interval [0,T]
    assuming the initial position of the earth is (x_e,0) and the initial
    position of the moon is (x_m,0).

    Parameters:
        T (int): The final time.
        x_e (float): The earth's initial x coordinate.
        x_m (float): The moon's initial x coordinate.
        omega_e (float): The earth's angular velocity.
        omega_m (float): The moon's angular velocity.
    """
    P_e0 = np.array([x_e, 0])
    P_m0 = np.array([x_m, 0])
    times = np.linspace(0,T,500)
    positions_E = np.array([rotate(P_e0, time*omega_e) for time in times])
    positions_M = np.array([rotate(P_m0-P_e0, times[t]*omega_m) + positions_E[t] for t in range(len(times))])
    plt.plot(positions_E[:,0], positions_E[:,1])
    plt.plot(positions_M[:,0], positions_M[:,1])
    plt.axis([min(positions_M[:,0])-1,max(positions_M[:,0])+1,min(positions_M[:,1])-1,max(positions_M[:,1])+1])
    plt.gca().set_aspect("equal")
    plt.show()


def random_vector(n):
    """Generate a random vector of length n as a list."""
    return [random() for i in range(n)]

def random_matrix(n):
    """Generate a random nxn matrix as a list of lists."""
    return [[random() for j in range(n)] for i in range(n)]

def matrix_vector_product(A, x):
    """Compute the matrix-vector product Ax as a list."""
    m, n = len(A), len(x)
    return [sum([A[i][k] * x[k] for k in range(n)]) for i in range(m)]

def matrix_matrix_product(A, B):
    """Compute the matrix-matrix product AB as a list of lists."""
    m, n, p = len(A), len(B), len(B[0])
    return [[sum([A[i][k] * B[k][j] for k in range(n)])
                                    for j in range(p) ]
                                    for i in range(m) ]

# Problem 3
def prob3():
    """Use time.time(), timeit.timeit(), or %timeit to time
    matrix_vector_product() and matrix-matrix-mult() with increasingly large
    inputs. Generate the inputs A, x, and B with random_matrix() and
    random_vector() (so each input will be nxn or nx1).
    Only time the multiplication functions, not the generating functions.

    Report your findings in a single figure with two subplots: one with matrix-
    vector times, and one with matrix-matrix times. Choose a domain for n so
    that your figure accurately describes the growth, but avoid values of n
    that lead to execution times of more than 1 minute.
    """
    domain = 2**np.arange(1,10)
    matrix_vector_times = []
    matrix_matrix_times = []
    # time the matrix products for various values of n
    for n in domain:
        A = random_matrix(n)
        B = random_matrix(n)
        x = random_vector(n)
        start_time = time.time()
        matrix_matrix_product(A,B)
        end_time = time.time()
        matrix_matrix_times.append(end_time - start_time)

        start_time = time.time()
        matrix_vector_product(A,x)
        end_time = time.time()
        matrix_vector_times.append(end_time - start_time)
    

    # plot the computed times
    plt.subplot(121)
    plt.plot(domain, matrix_vector_times)
    plt.title("Matrix-Vector Multiplication")
    plt.ylabel("Seconds")
    plt.xlabel("n")
    
    plt.subplot(122)
    plt.plot(domain, matrix_matrix_times)
    plt.title("Matrix-Matrix Multiplication")
    plt.ylabel("Seconds")
    plt.xlabel("n")

    plt.show()


# Problem 4
def prob4():
    """Time matrix_vector_product(), matrix_matrix_product(), and np.dot().

    Report your findings in a single figure with two subplots: one with all
    four sets of execution times on a regular linear scale, and one with all
    four sets of exections times on a log-log scale.
    """
    domain = 2**np.arange(1,10)
    matrix_vector_times = []
    matrix_matrix_times = []
    np_dot_matrix_times = []
    np_dot_vector_times = []
    # calcuate the times for each matrix multiplication for each n
    for n in domain:
        A = random_matrix(n)
        B = random_matrix(n)
        x = random_vector(n)
        start_time = time.time()
        matrix_matrix_product(A,B)
        end_time = time.time()
        matrix_matrix_times.append(end_time - start_time)

        start_time = time.time()
        matrix_vector_product(A,x)
        end_time = time.time()
        matrix_vector_times.append(end_time - start_time)

        A = np.array(A)
        B = np.array(B)
        x = np.array(x)

        start_time = time.time()
        A @ B
        end_time = time.time()
        np_dot_matrix_times.append(end_time - start_time)

        start_time = time.time()
        A @ x
        end_time = time.time()
        np_dot_vector_times.append(end_time - start_time)
    
    # plot the computed times
    plt.subplot(121)
    plt.plot(domain, matrix_matrix_times, 'k')
    plt.plot(domain, matrix_vector_times, 'b')
    plt.plot(domain, np_dot_matrix_times, 'r')
    plt.plot(domain, np_dot_vector_times, label = "NP Matrix-Vector")
    plt.legend(["Matrix-Matrix","Matrix-Vector","NP Matrix-Matrix","NP Matrix-Vector"])
    plt.title("Linear Plot")
    plt.xlabel('n')
    plt.ylabel("Seconds")

    plt.subplot(122)
    plt.loglog(domain, matrix_matrix_times, 'k')
    plt.loglog(domain, matrix_vector_times, 'b')
    plt.loglog(domain, np_dot_matrix_times, 'r')
    plt.loglog(domain, np_dot_vector_times, 'y')
    plt.legend(["Matrix-Matrix","Matrix-Vector","NP Matrix-Matrix","NP Matrix-Vector"])
    plt.title("Log-Log Plot")
    plt.xlabel('n')
    plt.ylabel("Seconds")
    
    plt.show()
    




if __name__ == "__main__":
    # data = np.load("horse.npy")
    # A = np.array([[1,1,1,1,1,1,1],[0,0,1,0,1,2,0]])
    # plt.plot(data[0], data[1], "b,")
    # plt.axis([-1,1,-1,1])
    # plt.gca().set_aspect("equal")
    # plt.show()
    # stretch_horse = rotate(data, np.pi)
    # plt.plot(stretch_horse[0], stretch_horse[1], "b,")
    # plt.axis([-1,1,-1,1])
    # plt.gca().set_aspect("equal")
    # plt.show()
    # solar_system(3*np.pi / 2, 10, 11, 1, 13)
    # prob3()
    prob4()
    pass