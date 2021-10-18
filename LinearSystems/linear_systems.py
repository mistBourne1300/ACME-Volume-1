# linear_systems.py
"""Volume 1: Linear Systems.
<Name>
<Class>
<Date>
"""

import numpy as np
from scipy import linalg as la
from scipy import sparse
from scipy.sparse import linalg as spla
from time import time
import matplotlib.pyplot as plt

# Problem 1
def ref(A):
    """Reduce the square matrix A to REF. You may assume that A is invertible
    and that a 0 will never appear on the main diagonal. Avoid operating on
    entries that you know will be 0 before and after a row operation.

    Parameters:
        A ((n,n) ndarray): The square invertible matrix to be reduced.

    Returns:
        ((n,n) ndarray): The REF of A.
    """
    for i in range(len(A) - 1):
        for j in range(i+1, len(A)):
            A[j, :] = A[j, :] - (A[j, i] / A[i, i]) * A[i, :]

    return A


# Problem 2
def lu(A):
    """Compute the LU decomposition of the square matrix A. You may
    assume that the decomposition exists and requires no row swaps.

    Parameters:
        A ((n,n) ndarray): The matrix to decompose.

    Returns:
        L ((n,n) ndarray): The lower-triangular part of the decomposition.
        U ((n,n) ndarray): The upper-triangular part of the decomposition.
    """
    m, n = np.shape(A)
    U=np.copy(A)
    L=np.identity(len(A))
    for j in range(n):
        for i in range(j+1, m):
            L[i,j] = U[i,j]/U[j,j]
            U[i,:] = U[i,:] - L[i,j] * U[j,:]
    return L,U # We've included the return values for you, though your function needs to define them correctly.
    raise NotImplementedError("Problem 2 Incomplete")

def forward_substitution(L,b):
  # Accepts a lower triangular square matrix L and a vector b, solves Ly=b for y.
  n = len(b)
  y = np.zeros(n)
  for i in range(n):
    y[i] = (b[i] - np.dot(y,L[i,:]))/L[i,i]
  return y


def back_substitution(U,y):
  # Accepts an upper triangular square matrix U and a vector b, solves Ux=b for x.
  n = len(y)
  x = np.zeros(n)
  for i in range(1, n+1):
    x[n-i] = (y[n-i] - np.dot(x, U[n-i,:]))/U[n-i, n-i]
  return x

# Problem 3
def solve(A, b):
    """Use the LU decomposition and back substitution to solve the linear
    system Ax = b. You may again assume that no row swaps are required.

    Parameters:
        A ((n,n) ndarray)
        b ((n,) ndarray)

    Returns:
        x ((m,) ndarray): The solution to the linear system.
    """
    L, U = lu(A)
    y = forward_substitution(L, b)
    x = back_substitution(U, y)
    return x



# Problem 4
def prob4():
    """Time different scipy.linalg functions for solving square linear systems.

    For various values of n, generate a random nxn matrix A and a random
    n-vector b using np.random.random(). Time how long it takes to solve the
    system Ax = b with each of the following approaches:

        1. Invert A with la.inv() and left-multiply the inverse to b.
        2. Use la.solve().
        3. Use la.lu_factor() and la.lu_solve() to solve the system with the
            LU decomposition.
        4. Use la.lu_factor() and la.lu_solve(), but only time la.lu_solve()
            (not the time it takes to do the factorization).

    Plot the system size n versus the execution times. Use log scales if
    needed.
    """
    sizes = 2**np.arange(1,13)
    inverse_solving = []
    la_solve = []
    lu_decomp_solve = []
    lu_decomp_solve_time_solving = []
    start_time, end_time = 0,0
    for n in sizes:
        A = np.random.random((n,n))
        b = np.random.random(n)

        # testing matrix inverse
        start_time = time()
        la.inv(A) @ b
        end_time = time()
        inverse_solving.append(end_time - start_time)

        # testing la.solve
        start_time = time()
        la.solve(A,b)
        end_time = time()
        la_solve.append(end_time - start_time)

        # testing la.lu_factor and la.lu_solve
        start_time = time()
        L,P = la.lu_factor(A)
        la.lu_solve((L,P),b)
        end_time = time()
        lu_decomp_solve.append(end_time - start_time)

        # testing only the la.lu_solve function
        
        start_time = time()
        la.lu_solve((L,P), b)
        end_time = time()
        lu_decomp_solve_time_solving.append(end_time - start_time)
    
    
    # print(inverse_solving, la_solve, lu_decomp_solve, lu_decomp_solve_time_solving,sep = "\n")

    plt.loglog(sizes, inverse_solving, 'k', base = 2)
    plt.loglog(sizes, la_solve, 'b', base = 2)
    plt.loglog(sizes, lu_decomp_solve, 'g', base = 2)
    plt.loglog(sizes, lu_decomp_solve_time_solving, 'r', base = 2)
    plt.xlabel("Matrix size")
    plt.ylabel("Time to Solve")
    plt.legend(["Inverse", "la_solve", "lu_factor + lu_solve", "lu_solve (alone)"])
    plt.show()




    


# Problem 5
def prob5(n):
    """Let I be the n × n identity matrix, and define
                    [B I        ]        [-4  1            ]
                    [I B I      ]        [ 1 -4  1         ]
                A = [  I . .    ]    B = [    1  .  .      ],
                    [      . . I]        [          .  .  1]
                    [        I B]        [             1 -4]
    where A is (n**2,n**2) and each block B is (n,n).
    Construct and returns A as a sparse matrix.

    Parameters:
        n (int): Dimensions of the sparse matrix B.

    Returns:
        A ((n**2,n**2) SciPy sparse matrix)
    """
    diagonals = [1,-4,1]
    offsets = [-1,0,1]

    B = sparse.diags(diagonals, offsets, shape = (n,n))
    Iden = sparse.diags([1],[0],shape=(n,n))
    A = sparse.block_diag([B]*n)
    A.setdiag(1,-n)
    A.setdiag(1,n)
    # plt.spy(A, markersize = 1)
    # plt.show()
    return A



# Problem 6
def prob6():
    """Time regular and sparse linear system solvers.

    For various values of n, generate the (n**2,n**2) matrix A described of
    prob5() and vector b of length n**2. Time how long it takes to solve the
    system Ax = b with each of the following approaches:

        1. Convert A to CSR format and use scipy.sparse.linalg.spsolve()
        2. Convert A to a NumPy array and use scipy.linalg.solve().

    In each experiment, only time how long it takes to solve the system (not
    how long it takes to convert A to the appropriate format). Plot the system
    size n**2 versus the execution times. As always, use log scales where
    appropriate and use a legend to label each line.
    """
    sizes = np.arange(2,75)
    CSR_timings = []
    NP_timings = []
    for n in sizes:
        # print(n)
        A = prob5(n)
        b = np.random.random(n**2) 

        # timing the optimized sparse matrix solver
        A_CSR = A.tocsr()
        start_time = time()
        spla.spsolve(A_CSR, b)
        end_time = time()
        CSR_timings.append(end_time - start_time)

        A_numpy = np.array(A.toarray())
        start_time = time()
        la.solve(A_numpy, b)
        end_time = time()
        NP_timings.append(end_time - start_time)

    plt.loglog(sizes, CSR_timings, 'k', base = 2)
    plt.loglog(sizes, NP_timings, 'r', base = 2)
    plt.xlabel("Matrix Size")
    plt.ylabel("Time to Solve")
    plt.legend(["CSR", "Regular"])
    plt.show()
    # plt.plot(sizes, CSR_timings)
    # plt.plot(sizes, NP_timings)
    # plt.legend(["CSR", "NP"])
    # plt.show()






    raise NotImplementedError("Problem 6 Incomplete")

if __name__ == "__main__":
    # A = np.array([  [1,5,8],
    #                 [2,3,7],
    #                 [4,10,3]])
    # # print(ref(A))
    # l,u = lu(A)
    # print(l,u,sep="\n\n")
    # b = np.array([100,45,37])
    # print(solve(A,b))
    # prob4()
    # print(prob5(10).toarray())
    prob4()
    prob6()
