# qr_decomposition.py
"""Volume 1: The QR Decomposition.
<Name>
<Class>
<Date>
"""

import numpy as np
import os
from scipy import linalg as la

sign = lambda x: 1 if x>=0 else -1

# Problem 1
def qr_gram_schmidt(A):
    """Compute the reduced QR decomposition of A via Modified Gram-Schmidt.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n.

    Returns:
        Q ((m,n) ndarray): An orthonormal matrix.
        R ((n,n) ndarray): An upper triangular matrix.
    """
    m, n = A.shape
    Q = A.copy()
    R = np.zeros((n,n))
    for i in range(n):
        R[i,i] = np.linalg.norm(Q[:,i])
        Q[:,i] = Q[:,i]/R[i,i]
        for j in range(i+1, n):
            R[i,j] = np.dot(Q[:,j], Q[:,i])
            Q[:,j] = Q[:,j] - R[i,j]*Q[:,i]
    return Q,R
    raise NotImplementedError("Problem 1 Incomplete")


# Problem 2
def abs_det(A):
    """Use the QR decomposition to efficiently compute the absolute value of
    the determinant of A.

    Parameters:
        A ((n,n) ndarray): A square matrix.

    Returns:
        (float) the absolute value of the determinant of A.
    """
    Q,R = qr_gram_schmidt(A)
    product = 1
    for i in range(len(R)):
        product *= R[i,i]
    return product
    raise NotImplementedError("Problem 2 Incomplete")


# Problem 3
def solve(A, b):
    """Use the QR decomposition to efficiently solve the system Ax = b.

    Parameters:
        A ((n,n) ndarray): An invertible matrix.
        b ((n, ) ndarray): A vector of length n.

    Returns:
        x ((n, ) ndarray): The solution to the system Ax = b.
    """
    Q, R = qr_gram_schmidt(A)
    y = Q.T @ b
    n = len(y)
    x = np.zeros(n)
    for i in range(1, n+1):
        x[n-i] = (y[n-i] - np.dot(x, R[n-i,:]))/R[n-i, n-i]
    return x
    raise NotImplementedError("Problem 3 Incomplete")


# Problem 4
def qr_householder(A):
    """Compute the full QR decomposition of A via Householder reflections.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n.

    Returns:
        Q ((m,m) ndarray): An orthonormal matrix.
        R ((m,n) ndarray): An upper triangular matrix.
    """
    m, n = A.shape
    R = A.copy()
    Q = np.identity(m)
    for k in range(n):
        u = R[k:,k].copy()
        u[0] = u[0] + sign(u[0]) * np.linalg.norm(u)
        u = u / np.linalg.norm(u)
        R[k:, k:] = R[k:, k:] - 2 * np.outer(u, u.T @ R[k:,k:])
        Q[k:,:] = Q[k:,:] - 2 * np.outer(u, u.T @ Q[k:,:])
        print("R:\n", R)
    return Q.T, R
    raise NotImplementedError("Problem 4 Incomplete")


# Problem 5
def hessenberg(A):
    """Compute the Hessenberg form H of A, along with the orthonormal matrix Q
    such that A = QHQ^T.

    Parameters:
        A ((n,n) ndarray): An invertible matrix.

    Returns:
        H ((n,n) ndarray): The upper Hessenberg form of A.
        Q ((n,n) ndarray): An orthonormal matrix.
    """
    m, n = A.shape
    H = A.copy()
    Q = np.identity(m)
    for k in range(n-2):
        u = H[k+1:, k].copy()
        u[0] = u[0] + sign(u[0]) * np.linalg.norm(u)
        u = u / np.linalg.norm(u)
        # u_t = u.T@H[k+1:, k:]
        # np.outer(u,u_t)
        # H[k+1:,k:] - 2*np.outer(u, u.T@H[k+1:, k:])
        H[k+1:,k:] = H[k+1:,k:] - 2*np.outer(u, u.T@H[k+1:, k:])
        H[:, k+1:] = H[:,k+1:] - 2 * np.outer(H[:, k+1:] @ u, u.T)
        Q[k+1:,:] = Q[k+1:,:] - 2 * np.outer(u, u.T @ Q[k+1:,:])
        print("H:\n", H)
    
    return H, Q.T
    raise NotImplementedError("Problem 5 Incomplete")

if __name__ == "__main__":
    A = np.random.random((4,4))
    print("A:\n", A)
    Q,R = qr_gram_schmidt(A)
    print("\n\n")
    print("Q:\n", Q)
    print("\n\n")
    print("R:\n", R)

    print("R upper tri: ", np.allclose(np.triu(R), R))
    print("Q orthonormal: ", np.allclose(Q.T @ Q, np.identity(len(Q))))
    print("Q @ R = A: ", np.allclose(Q@R, A))

    print("Determinant works: ", np.allclose(abs(np.linalg.det(A)), abs_det(A)))

    print("testing Ax=b solve:")
    b = np.random.random(4)
    x = solve(A,b)
    print("correctly solved: ", np.allclose(A@x, b))

    print("testing householder: ")

    print("A:\n", A)
    Q,R = qr_householder(A)
    print("\n\n")
    print("Q:\n", Q)
    print("\n\n")
    print("R:\n", R)

    print("R upper tri: ", np.allclose(np.triu(R), R))
    print("Q orthonormal: ", np.allclose(Q.T @ Q, np.identity(len(Q))))
    print("Q @ R = A: ", np.allclose(Q@R, A))

    print("testing hessenberg:")

    print("A:\n", A)
    H, Q = hessenberg(A)
    print("\n\n")
    print("H:\n", H)
    print("\n\n")
    print("Q:\n", Q)

    print("Q orthonormal: ", np.allclose(Q.T @ Q, np.identity(len(Q))))
    print("A = Q@H@Q.T: ", np.allclose(A, Q@H@Q.T))

