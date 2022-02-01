# condition_stability.py
"""Volume 1: Conditioning and Stability.
<Name>
<Class>
<Date>
"""

import os
import numpy as np
import sympy as sy
from scipy import linalg as la
import matplotlib.pyplot as plt


# Problem 1
def matrix_cond(A):
    """Calculate the condition number of A with respect to the 2-norm."""
    sing_vals = la.svdvals(A)
    if sing_vals[-1] == 0:
        return np.inf
    return sing_vals[0]/sing_vals[-1]


# Problem 2
def prob2():
    """Randomly perturb the coefficients of the Wilkinson polynomial by
    replacing each coefficient c_i with c_i*r_i, where r_i is drawn from a
    normal distribution centered at 1 with standard deviation 1e-10.
    Plot the roots of 100 such experiments in a single figure, along with the
    roots of the unperturbed polynomial w(x).

    Returns:
        (float) The average absolute condition number.
        (float) The average relative condition number.
    """
    w_roots = np.arange(1, 21)

    # Get the exact Wilkinson polynomial coefficients using SymPy.
    x, i = sy.symbols('x i')
    w = sy.poly_from_expr(sy.product(x-i, (i, 1, 20)))[0]
    w_coeffs = np.array(w.all_coeffs())
    real_roots = np.roots(w_coeffs)
    plt.plot(real_roots.real, real_roots.imag, 'ro')

    cond_numbers = []
    rel_cond_numbers = []
    for i in range(100):
        pirates = np.random.normal(loc = 1, scale = 1e-10, size = len(w_coeffs))
        roots = np.roots(w_coeffs*pirates)
        plt.plot(roots.real, roots.imag, 'k,')
        cond_numbers.append(la.norm(roots - real_roots, np.inf)/la.norm(pirates, np.inf))
        rel_cond_numbers.append(cond_numbers[-1]*la.norm(w_coeffs, np.inf)/la.norm(real_roots, np.inf))
    plt.show()
    return np.mean(cond_numbers), np.mean(rel_cond_numbers)


# Helper function
def reorder_eigvals(orig_eigvals, pert_eigvals):
    """Reorder the perturbed eigenvalues to be as close to the original eigenvalues as possible.
    
    Parameters:
        orig_eigvals ((n,) ndarray) - The eigenvalues of the unperturbed matrix A
        pert_eigvals ((n,) ndarray) - The eigenvalues of the perturbed matrix A+H
        
    Returns:
        ((n,) ndarray) - the reordered eigenvalues of the perturbed matrix
    """
    n = len(pert_eigvals)
    sort_order = np.zeros(n).astype(int)
    dists = np.abs(orig_eigvals - pert_eigvals.reshape(-1,1))
    for _ in range(n):
        index = np.unravel_index(np.argmin(dists), dists.shape)
        sort_order[index[0]] = index[1]
        dists[index[0],:] = np.inf
        dists[:,index[1]] = np.inf
    return pert_eigvals[sort_order]

# Problem 3
def eig_cond(A):
    """Approximate the condition numbers of the eigenvalue problem at A.

    Parameters:
        A ((n,n) ndarray): A square matrix.

    Returns:
        (float) The absolute condition number of the eigenvalue problem at A.
        (float) The relative condition number of the eigenvalue problem at A.
    """
    reals = np.random.normal(0, 1e-10, A.shape)
    imags = np.random.normal(0, 1e-10, A.shape)
    H = reals + 1j*imags
    _A_ = A + H
    orig_eigvals = la.eigvals(A)
    pert_eigvals  = reorder_eigvals(orig_eigvals, la.eigvals(_A_))
    kappa_hat = la.norm(orig_eigvals - pert_eigvals, ord = 2)/la.norm(H, ord = 2)
    return kappa_hat, kappa_hat * la.norm(A, ord = 2)/la.norm(orig_eigvals, ord = 2)


# Problem 4
def prob4(domain=[-100, 100, -100, 100], res=50):
    """Create a grid [x_min, x_max] x [y_min, y_max] with the given resolution. For each
    entry (x,y) in the grid, find the relative condition number of the
    eigenvalue problem, using the matrix   [[1, x], [y, 1]]  as the input.
    Use plt.pcolormesh() to plot the condition number over the entire grid.

    Parameters:
        domain ([x_min, x_max, y_min, y_max]):
        res (int): number of points along each edge of the grid.
    """
    exxes = np.linspace(domain[0], domain[1], res)
    whys = np.linspace(domain[2], domain[3], res)
    EXXES, WHYS = np.meshgrid(exxes, whys)
    cond_numbers = np.zeros((res,res))
    for i,x in enumerate(exxes):
        for j,y in enumerate(whys):
            cond_numbers[i][j] = eig_cond(np.array([[1,x],[y,1]]))[1]
    plt.pcolormesh(EXXES, WHYS, cond_numbers, cmap = 'inferno')
    plt.show()


# Problem 5
def prob5(n):
    """Approximate the data from "stability_data.npy" on the interval [0,1]
    with a least squares polynomial of degree n. Solve the least squares
    problem using the normal equation and the QR decomposition, then compare
    the two solutions by plotting them together with the data. Return
    the mean squared error of both solutions, ||Ax-b||_2.

    Parameters:
        n (int): The degree of the polynomial to be used in the approximation.

    Returns:
        (float): The forward error using the normal equations.
        (float): The forward error using the QR decomposition.
    """
    xk, yk = np.load("stability_data.npy").T
    A = np.vander(xk, n+1)

    x_unstable = la.inv(A.T@A)@A.T@yk
    domain = np.linspace(0,1, 1000)
    plt.plot(domain, np.polyval(x_unstable, domain), 'k')


    Q,R = la.qr(A, mode ='economic')
    x_stable = la.solve_triangular(R, Q.T@yk)

    plt.plot(domain, np.polyval(x_stable,domain), 'goldenrod')

    plt.plot(xk,yk,'ro', markersize = 1)
    plt.legend(['unstable', 'stable', 'original points'])
    plt.title("unstable vs stable least squares")
    plt.show()

    return la.norm(A@x_unstable - yk), la.norm(A@x_stable - yk)


# Problem 6
def prob6():
    """For n = 5, 10, ..., 50, compute the integral I(n) using SymPy (the
    true values) and the subfactorial formula (may or may not be correct).
    Plot the relative forward error of the subfactorial formula for each
    value of n. Use a log scale for the y-axis.
    """
    enns = np.arange(1,11)*5
    for_errs = []
    for n in enns:
        n0 = int(n)
        x = sy.symbols('x')
        expr = x**n0 * sy.exp(x-1)
        true_I = sy.integrate(expr, (x,0,1))

        fake_I = (-1)**n * (sy.subfactorial(n0) - sy.factorial(n0)/np.e)

        for_errs.append(np.abs(true_I - fake_I)/np.abs(true_I))

    plt.plot(enns, for_errs, 'goldenrod')
    plt.yscale('log')
    plt.title("Errors")
    plt.xlabel("n value")
    plt.ylabel("Error")
    plt.show()


if __name__ == "__main__":
    # problem 1
    # A = np.random.rand(4,4)
    # print(A)
    # Q,R = la.qr(A)
    # print(matrix_cond(Q))

    # problem 2
    # print(prob2())

    # problem 3
    # A = np.random.rand(4,4)
    # print(eig_cond(A))

    # problem 4
    # prob4(res = 200)

    # problem 5
    # os.chdir("/Users/chase/Desktop/Math345Volume1/byu_vol1/Conditioning_Stability")
    # print(prob5(5))

    # problem 6
    # prob6()
    pass