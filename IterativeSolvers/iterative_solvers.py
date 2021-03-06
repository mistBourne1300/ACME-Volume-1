# iterative_solvers.py
"""Volume 1: Iterative Solvers.
<Name>
<Class>
<Date>
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse


# Helper function
def diag_dom(n, num_entries=None, as_sparse=False):
    """Generate a strictly diagonally dominant (n, n) matrix.
    Parameters:
        n (int): The dimension of the system.
        num_entries (int): The number of nonzero values.
            Defaults to n^(3/2)-n.
        as_sparse: If True, an equivalent sparse CSR matrix is returned.
    Returns:
        A ((n,n) ndarray): A (n, n) strictly diagonally dominant matrix.
    """
    if num_entries is None:
        num_entries = int(n**1.5) - n
    A = sparse.dok_matrix((n,n))
    rows = np.random.choice(n, size=num_entries)
    cols = np.random.choice(n, size=num_entries)
    data = np.random.randint(-4, 4, size=num_entries)
    for i in range(num_entries):
        A[rows[i], cols[i]] = data[i]
    B = A.tocsr()          # convert to row format for the next step
    for i in range(n):
        A[i,i] = abs(B[i]).sum() + 1
    return A.tocsr() if as_sparse else A.toarray()

# Problems 1 and 2
def jacobi(A, b, tol=1e-8, maxiter=100, plot=False):
    """Calculate the solution to the system Ax = b via the Jacobi Method.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        b ((n ,) ndarray): A vector of length n.
        tol (float): The convergence tolerance.
        maxiter (int): The maximum number of iterations to perform.

    Returns:
        ((n,) ndarray): The solution to system Ax = b.
    """
    # get the diagonals and create a the zero vector
    d = np.diag(A)
    x0=np.zeros(b.shape)
    errs = []
    for i in range(maxiter):
        # update the vector
        x1 = x0 + (b-A@x0)/d
        # if we're plotting, add to the errors vector
        if plot:
            errs.append(np.linalg.norm(A@x1 - b, ord = np.inf))
        # if the norm is within tolerance, we're done, plot and return 
        if np.linalg.norm(x1-x0,ord = np.inf)<tol:
            if plot:
                plt.semilogy(errs)
                plt.title("convergence of Jacobi Method")
                plt.xlabel("Iteration")
                plt.ylabel("Absolute Error of Approximation")
                plt.show()

            return x1
        
        # update x0
        x0 = x1
    
    # plot and return anyway
    if plot:
        plt.semilogy(errs)
        plt.title("convergence of Jacobi Method")
        plt.xlabel("Iteration")
        plt.ylabel("Absolute Error of Approximation")
        plt.show()
    return x1


# Problem 3
def gauss_seidel(A, b, tol=1e-8, maxiter=100, plot=False):
    """Calculate the solution to the system Ax = b via the Gauss-Seidel Method.

    Parameters:
        A ((n, n) ndarray): A square matrix.
        b ((n, ) ndarray): A vector of length n.
        tol (float): The convergence tolerance.
        maxiter (int): The maximum number of iterations to perform.
        plot (bool): If true, plot the convergence rate of the algorithm.

    Returns:
        x ((n,) ndarray): The solution to system Ax = b.
    """
    # zero vector
    x0 = np.zeros(b.shape)
    errs = []
    for i in range(maxiter):
        # copycat vector
        x1 = x0.copy()
        #update the vector one index at a time
        for j in range(len(x1)):
            x1[j] = x1[j] + (b[j] - A[j,:]@x1)/A[j,j]
        
        # add to errors if we're plotting
        if plot:
            errs.append(np.linalg.norm(A@x1 - b,ord = np.inf))
        # if the norm is within tolerance, plot and return
        if np.linalg.norm(x1 - x0, ord = np.inf) < tol:
            if plot:
                plt.semilogy(errs)
                plt.title("Convergence of Gauss-Seidel Method")
                plt.xlabel("Iteration")
                plt.ylabel("Absolute Error of Approximation")
                plt.show()
            return x1
        
        # update x0
        x0 = x1

    # plot and return anyway
    if plot:
        plt.semilogy(errs)
        plt.title("convergence of Jacobi Method")
        plt.xlabel("Iteration")
        plt.ylabel("Absolute Error of Approximation")
        plt.show()
    return x1


# Problem 4
def gauss_seidel_sparse(A, b, tol=1e-8, maxiter=100):
    """Calculate the solution to the sparse system Ax = b via the Gauss-Seidel
    Method.

    Parameters:
        A ((n, n) csr_matrix): A (n, n) sparse CSR matrix.
        b ((n, ) ndarray): A vector of length n.
        tol (float): The convergence tolerance.
        maxiter (int): the maximum number of iterations to perform.

    Returns:
        x ((n,) ndarray): The solution to system Ax = b.
    """
    # zero vector
    x0 = np.zeros(b.shape)
    for i in range(maxiter):
        # copycat vector
        x1 = x0.copy()
        for j in range(len(x1)):
            # get the rowstart and rowend
            rowstart = A.indptr[j]
            rowend = A.indptr[j+1]

            # update the vector one component at a time
            Aix = A.data[rowstart:rowend] @ x1[A.indices[rowstart:rowend]]
            x1[j] = x1[j] + (b[j] - Aix)/A[j,j]
        
        # if norm is within tol, we're done
        if np.linalg.norm(x1 - x0, ord = np.inf) < tol:
            return x1
        
        # update x0
        x0 = x1
    
    # return anyway
    return x1


# Problem 5
def sor(A, b, omega, tol=1e-8, maxiter=100):
    """Calculate the solution to the system Ax = b via Successive Over-
    Relaxation.

    Parameters:
        A ((n, n) csr_matrix): A (n, n) sparse matrix.
        b ((n, ) Numpy Array): A vector of length n.
        omega (float in [0,1]): The relaxation factor.
        tol (float): The convergence tolerance.
        maxiter (int): The maximum number of iterations to perform.

    Returns:
        ((n,) ndarray): The solution to system Ax = b.
        (bool): Whether or not Newton's method converged.
        (int): The number of iterations computed.
    """
    # zero vector
    x0 = np.zeros(b.shape)
    for i in range(maxiter):
        # copycat vector
        x1 = x0.copy()
        for j in range(len(x1)):
            # get rowstart and rowend
            rowstart = A.indptr[j]
            rowend = A.indptr[j+1]

            # update the vector one component at a time (omega included)
            Aix = A.data[rowstart:rowend] @ x1[A.indices[rowstart:rowend]]
            x1[j] = x1[j] + omega*(b[j] - Aix)/A[j,j]
        
        # if the norm is less than tol, we're done
        if np.linalg.norm(x1 - x0, ord = np.inf) < tol:
            return x1, True, i
        
        # update x0
        x0 = x1
    
    # return anyway
    return x1, False, i


# Problem 6
def hot_plate(n, omega, tol=1e-8, maxiter=100, plot=False):
    """Generate the system Au = b and then solve it using sor().
    If show is True, visualize the solution with a heatmap.

    Parameters:
        n (int): Determines the size of A and b.
            A is (n^2, n^2) and b is one-dimensional with n^2 entries.
        omega (float in [0,1]): The relaxation factor.
        tol (float): The iteration tolerance.
        maxiter (int): The maximum number of iterations.
        plot (bool): Whether or not to visualize the solution.

    Returns:
        ((n^2,) ndarray): The 1-D solution vector u of the system Au = b.
        (bool): Whether or not Newton's method converged.
        (int): The number of computed iterations in SOR.
    """
    # helper function to generate the A matrix
    def generate_A():
        diagonals = [1,-4,1]
        offsets = [-1,0,1]
        B = sparse.diags(diagonals, offsets, shape = (n,n))
        Iden = sparse.diags([1],[0],shape=(n,n))
        A = sparse.block_diag([B]*n)
        A.setdiag(1,-n)
        A.setdiag(1,n)
        return A
    
    # create a small piece of the b vector, then us np.tile to make the rest of it
    smallb = np.zeros(n)
    smallb[0] = -100
    smallb[-1] = -100
    b = np.tile(smallb,n)
    
    # get the u vector, the convergent state, and the iteration count from the sor method
    u, converged, iter = sor(generate_A().tocsr(),b,omega=omega, tol = tol, maxiter = maxiter)
    
    # plot and return
    if plot:
        U = u.reshape((n,n))
        plt.pcolormesh(U,cmap='coolwarm')
        plt.show()
    return u, converged, iter


# Problem 7
def prob7():
    """Run hot_plate() with omega = 1, 1.05, 1.1, ..., 1.9, 1.95, tol=1e-2,
    and maxiter = 1000 with A and b generated with n=20. Plot the iterations
    computed as a function of omega.
    """
    # all the omegas we are testing
    omegas = np.arange(1,2,step=.05)
    # vector to hold the iterations
    num_iters = []
    for w in omegas:
        # get the iterations used for the particular omega and append to num_iters vector
        u,converged,iter = hot_plate(20, w, tol = 1e-2, maxiter = 1000)
        num_iters.append(iter)
    
    # plot the iterations vs the omegas
    plt.plot(omegas, num_iters)
    plt.title("iterations vs omega")
    plt.xlabel("omega")
    plt.ylabel("iterations")
    plt.show()

    # return the minimum omega value
    return omegas[np.argmin(num_iters)]


if __name__ == "__main__":
    # n = 100
    # A = diag_dom(n, as_sparse=True)
    # b = np.random.random(n)
    # gauss_seidel(A,b,plot=True)
    # x, converge,iter = sor(A,b,1.3)
    # print(np.allclose(A@x,b))
    # print(converge, iter)
    
    # print(hot_plate(20, 1, tol = 1e-2, maxiter = 1000, plot = True))
    # print(prob7())
    pass
