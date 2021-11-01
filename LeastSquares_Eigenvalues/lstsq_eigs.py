# lstsq_eigs.py
"""Volume 1: Least Squares and Computing Eigenvalues.
<Name>
<Class>
<Date>
"""

# (Optional) Import functions from your QR Decomposition lab.
# import sys
# sys.path.insert(1, "../QR_Decomposition")
# from qr_decomposition import qr_gram_schmidt, qr_householder, hessenberg

import numpy as np
from numpy.ma.core import power
from scipy import linalg as la
from matplotlib import pyplot as plt
import os
import cmath


# Problem 1
def least_squares(A, b):
    """Calculate the least squares solutions to Ax = b by using the QR
    decomposition.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n <= m.
        b ((m, ) ndarray): A vector of length m.

    Returns:
        x ((n, ) ndarray): The solution to the normal equations.
    """
    Q,R = la.qr(A, mode='economic')
    xhat = la.solve_triangular(R, Q.T @ b)
    return xhat

    raise NotImplementedError("Problem 1 Incomplete")

# Problem 2
def line_fit():
    """Find the least squares line that relates the year to the housing price
    index for the data in housing.npy. Plot both the data points and the least
    squares line.
    """
    housing = np.load("housing.npy")
    A = np.vander(housing[:,0], 2)
    b = housing[:,1]
    xhat = least_squares(A,b)
    plt.scatter(housing[:,0], housing[:,1], s=1)
    domain = np.linspace(0,16)
    plt.plot(domain, xhat[0]*domain + xhat[1], 'k')
    plt.show()



# Problem 3
def polynomial_fit():
    """Find the least squares polynomials of degree 3, 6, 9, and 12 that relate
    the year to the housing price index for the data in housing.npy. Plot both
    the data points and the least squares polynomials in individual subplots.
    """
    housing = np.load("housing.npy")
    domain = np.linspace(0,16)
    plt.subplot(221)
    plt.title("Raw Housing Data")
    plt.xlabel("Year")
    plt.ylabel("Housing Index")
    plt.scatter(housing[:,0], housing[:,1], s=1)
    b = housing[:,1]
    # degree three
    A = np.vander(housing[:,0], 4)
    xhat = la.lstsq(A,b)[0]
    plt.subplot(222)
    plt.title("3rd Degree Best Fit")
    plt.xlabel("Year")
    plt.ylabel("Housing Index")
    plt.plot(domain, xhat[0]*(domain**3) + xhat[1]*(domain**2) + xhat[2]*domain + xhat[3], 'k')

    # degree 6
    A = np.vander(housing[:,0], 7)
    xhat = la.lstsq(A,b)[0]
    plt.subplot(223)
    plt.title("6th Degree Best Fit")
    plt.xlabel("Year")
    plt.ylabel("Housing Index")
    plt.plot(domain, xhat[0]*(domain**6) + xhat[1]*(domain**5) + xhat[2]*(domain**4) + xhat[3]*(domain**3) + xhat[4]*(domain**2) + xhat[5]*(domain) + xhat[6], 'b')

    # degree 12
    A = np.vander(housing[:,0], 13)
    xhat = la.lstsq(A,b)[0]
    plt.subplot(224)
    plt.title("12th Degree Best Fit")
    plt.xlabel("Year")
    plt.ylabel("Housing Index")
    plt.plot(domain, xhat[0]*(domain**12) + xhat[1]*(domain**11) + xhat[2]*(domain**10) + xhat[3]*(domain**9) + xhat[4]*(domain**8) + xhat[5]*(domain**7) + xhat[6]*(domain**6) + xhat[7]*(domain**5) + xhat[8]*(domain**4) + xhat[9]*(domain**3) + xhat[10]*(domain**2) + xhat[11]*domain + xhat[12], 'r')
    plt.show()


def plot_ellipse(a, b, c, d, e):
    """Plot an ellipse of the form ax^2 + bx + cxy + dy + ey^2 = 1."""
    theta = np.linspace(0, 2*np.pi, 200)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    A = a*(cos_t**2) + c*cos_t*sin_t + e*(sin_t**2)
    B = b*cos_t + d*sin_t
    r = (-B + np.sqrt(B**2 + 4*A)) / (2*A)

    plt.plot(r*cos_t, r*sin_t)
    plt.gca().set_aspect("equal", "datalim")


# Problem 4
def ellipse_fit():
    """Calculate the parameters for the ellipse that best fits the data in
    ellipse.npy. Plot the original data points and the ellipse together, using
    plot_ellipse() to plot the ellipse.
    """
    ellipse_points = np.load("ellipse.npy")
    A = np.array([[p[0]**2, p[0], p[0]*p[1], p[1], p[1]**2] for p in ellipse_points])
    xhat = la.lstsq(A, [1 for i in A])[0]
    plot_ellipse(xhat[0], xhat[1], xhat[2], xhat[3], xhat[4])
    
    plt.scatter(ellipse_points[:,0], ellipse_points[:,1], s=1)
    plt.show()


# Problem 5
def power_method(A, N=20, tol=1e-12):
    """Compute the dominant eigenvalue of A and a corresponding eigenvector
    via the power method.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The maximum number of iterations.
        tol (float): The stopping tolerance.

    Returns:
        (float): The dominant eigenvalue of A.
        ((n,) ndarray): An eigenvector corresponding to the dominant
            eigenvalue of A.
    """
    m,n = A.shape
    x0 = np.random.random(n)
    x0 = x0/np.linalg.norm(x0)
    for k in range(N):
        x_prev = x0
        x0 = A@x0
        x0 = x0/np.linalg.norm(x0)
        if(np.linalg.norm(x0-x_prev) < tol):
            return x0.T@A@x0, x0
    return x0.T@A@x0, x0
    raise NotImplementedError("Problem 5 Incomplete")


# Problem 6
def qr_algorithm(A, N=50, tol=1e-12):
    """Compute the eigenvalues of A via the QR algorithm.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The number of iterations to run the QR algorithm.
        tol (float): The threshold value for determining if a diagonal S_i
            block is 1x1 or 2x2.

    Returns:
        ((n,) ndarray): The eigenvalues of A.
    """
    m,n = A.shape
    S = la.hessenberg(A)
    for k in range(N):
        Q,R = la.qr(S)
        S = R@Q
    eigs = []
    i = 0
    while i<n:
        if i == n-1:
            eigs.append(S[i,i])
        elif S[i+1,i] < tol: # checks the entry one row down from the diagonal
            eigs.append(S[i,i])
        else:
            a = S[i,i]
            b = S[i,i+1]
            c = S[i+1,i]
            d = S[i+1,i+1]
            lam1 = (a+d + cmath.sqrt((a+d)**2 - 4*(a*d - b*c))) / 2
            lam2 = (a+d - cmath.sqrt((a+d)**2 - 4*(a*d - b*c))) / 2
            eigs.append([lam1, lam2])
            i+=1
        i+=1
    return eigs

    raise NotImplementedError("Problem 6 Incomplete")

if __name__ == "__main__":
    os.chdir("/Users/chase/Desktop/Math345Volume1/byu_vol1/LeastSquares_Eigenvalues")
    ##### prob 1 #####

    A = np.random.random((100,9))
    print("A: ", A.shape, "\n", A)
    b = np.random.random(100)
    print("b:\n", b)

    xhat = least_squares(A,b)
    print("xhat:\n", xhat)
    print(A@xhat, "\nnorm:", np.linalg.norm(A@xhat-b))
    line_fit()
    print("done with line fit")
    polynomial_fit()
    ellipse_fit()
    
    
    A = np.random.random((10,10))
    # print(A)
    eigs, vecs = la.eig(A)
    print(eigs)
    # index = np.argmax(eigs)
    # e,v = power_method(A)
    # print(np.allclose(eigs[index], e))
    # print(f'{e}, {v}')

    print(qr_algorithm(A))
    pass