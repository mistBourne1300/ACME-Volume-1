# solutions.py
"""Volume 1: The SVD and Image Compression. Solutions File."""


from numpy.linalg import svd
from scipy import linalg as la
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

# Problem 1
def compact_svd(A, tol=1e-6):
    """Compute the truncated SVD of A.

    Parameters:
        A ((m,n) ndarray): The matrix (of rank r) to factor.
        tol (float): The tolerance for excluding singular values.

    Returns:
        ((m,r) ndarray): The orthonormal matrix U in the SVD.
        ((r,) ndarray): The singular values of A as a 1-D array.
        ((r,n) ndarray): The orthonormal matrix V^H in the SVD.
    """

    eigvals, eigvects = la.eig(A.T@A) 
    eigvals = np.real(eigvals) # get just the real parts of the eigenvalues
    sortindex = np.argsort(eigvals) # get the array to sort the eigenvalues
    eigvals = np.array([eigvals[index] for index in sortindex[::-1]]) # reassign eigvals to the argsorted version

    r = np.sum(eigvals > tol)
    sing = np.array([np.sqrt(lam) for lam in eigvals[:r]])

    eigvects = np.array([eigvects[index] for index in sortindex[::-1]]) # sort the vectors the same way the values were sorted
    r = np.sum(sing > tol)
    sigma1 = sing
    V1 = eigvects[:,:r]
    U1 = A@V1 / sigma1
    return np.real(U1), sigma1, np.real(V1.T)

# Problem 2
def visualize_svd(A):
    """Plot the effect of the SVD of A as a sequence of linear transformations
    on the unit circle and the two standard basis vectors.
    """
    U, Sig, Vh = la.svd(A)
    E = np.array([  [1,0,0],
                    [0,0,1]])
    Sig = la.diagsvd(Sig, A.shape[0], A.shape[1])

    theta = np.linspace(0,2*np.pi, 200)
    unit_circle = np.array([np.cos(theta), np.sin(theta)])
    plt.subplot(221)
    plt.plot(unit_circle[0,:], unit_circle[1,:])
    plt.plot([1,0,0], [0,0,1])
    plt.axis("equal")

    plt.subplot(222)
    VHS = Vh@unit_circle
    VHE = Vh@E
    plt.plot(VHS[0,:], VHS[1,:])
    plt.plot(VHE[0,:], VHE[1,:])
    plt.axis("equal")

    plt.subplot(223)
    SVHS = Sig@VHS
    SVHE = Sig@VHE
    plt.plot(SVHS[0,:], SVHS[1,:])
    plt.plot(SVHE[0,:], SVHE[1,:])
    plt.axis("equal")

    plt.subplot(224)
    USVHS = U@SVHS
    USVHE = U@SVHE
    plt.plot(USVHS[0,:], USVHS[1,:])
    plt.plot(USVHE[0,:], USVHE[1,:])
    plt.axis("equal")

    plt.suptitle("SVD transformations on a unit circle")
    plt.show()


# Problem 3
def svd_approx(A, s):
    """Return the best rank s approximation to A with respect to the 2-norm
    and the Frobenius norm, along with the number of bytes needed to store
    the approximation via the truncated SVD.

    Parameters:
        A ((m,n), ndarray)
        s (int): The rank of the desired approximation.

    Returns:
        ((m,n), ndarray) The best rank s approximation of A.
        (int) The number of entries needed to store the truncated SVD.
    """
    if np.linalg.matrix_rank(A) < s:
        raise ValueError("s is more than the matrix rank")
    U, S, Vh = la.svd(A, full_matrices = False)
    Sig = la.diagsvd(S, A.shape[0], A.shape[1])
    Uhat = U[:,:s]
    Shat = Sig[:s,:s]
    Vhat = Vh[:s,:]

    return Uhat@Shat@Vhat, Uhat.size + s + Vhat.size



# Problem 4
def lowest_rank_approx(A, err):
    """Return the lowest rank approximation of A with error less than 'err'
    with respect to the matrix 2-norm, along with the number of bytes needed
    to store the approximation via the truncated SVD.

    Parameters:
        A ((m, n) ndarray)
        err (float): Desired maximum error.

    Returns:
        A_s ((m,n) ndarray) The lowest rank approximation of A satisfying
            ||A - A_s||_2 < err.
        (int) The number of entries needed to store the truncated SVD.
    """
    rankA = np.linalg.matrix_rank(A)
    U,S,Vh = la.svd(A, full_matrices = False)
    Sig = la.diagsvd(S, A.shape[0], A.shape[1])

    s=0
    while S[s] > err:
        s+=1
        if(s>=rankA):
            raise ValueError("No approximation exists within error tolerance")
    
    Uhat = U[:,:s]
    Shat = Sig[:s,:s]
    Vhat = Vh[:s,:]

    return Uhat@Shat@Vhat, Uhat.size + s + Vhat.size


# Problem 5
def compress_image(filename, s):
    """Plot the original image found at 'filename' and the rank s approximation
    of the image found at 'filename.' State in the figure title the difference
    in the number of entries used to store the original image and the
    approximation.

    Parameters:
        filename (str): Image file path.
        s (int): Rank of new image.
    """
    image = imageio.imread(filename)/255
    
    
    # color image
    if len(image.shape) == 3:
        plt.subplot(121)
        plt.imshow(image)
        plt.axis("off")
        plt.subplot(122)

        R = image[:,:,0]
        G = image[:,:,1]
        B = image[:,:,2]
        Rcomp, rsize = svd_approx(R, s)
        Gcomp, gsize = svd_approx(G, s)
        Bcomp, bsize = svd_approx(B, s)

        RGB = np.clip(np.dstack((Rcomp,Gcomp,Bcomp)), 0, 1)
        plt.imshow(RGB)
        plt.axis("off")
        plt.suptitle(f'Compressed image with s={s}. Entry diff: {image.size - rsize - gsize - bsize} indices saved')

    
    # greyscale image
    else: 
        plt.subplot(121)
        plt.imshow(image, cmap = 'gray')
        plt.axis("off")
        plt.subplot(122)
        compressed, size = svd_approx(image, s)
        compressed = np.clip(compressed, 0, 1)
        plt.imshow(compressed, cmap = 'gray')
        plt.axis("off")
        plt.suptitle(f'Compressed image with s={s}. Entry diff: {image.size - size} indices saved')



    plt.show()



if __name__ == "__main__":
    A = np.random.random((5,10))
    # A = np.array([  [3., 9., 1., 1., 3., 7., 3., 3., 2., 2.],
    #                 [6., 9., 2., 8., 8., 5., 8., 3., 2., 7.],
    #                 [2., 2., 1., 2., 7., 8., 9., 4., 2., 3.],
    #                 [1., 8., 4., 2., 9., 3., 2., 4., 2., 5.],
    #                 [8., 8., 4., 7., 8., 3., 9., 8., 4., 9.]])
    # print(A)

    # u1, sigma1, v1h = compact_svd(A)
    # Sigma = np.diag(sigma1)
    # print(u1@Sigma@v1h)

    # A = np.array([[3,1],[1,3]])
    # visualize_svd(A)

    # A = np.random.randint(0,10,size =(40,45))
    # print(f'rank: {np.linalg.matrix_rank(A)}')
    # print(f'A: {A}\n\n')
    # SVD_approx, size = svd_approx(A,10)
    # print(np.linalg.matrix_rank(SVD_approx))
    # print(f'SVD_approx: {SVD_approx}')
    # print(f'size: {size}')

    os.chdir("/Users/chase/Desktop/Math345Volume1/byu_vol1/SVD_ImageCompression")
    hubble = "hubble.jpg"
    nate = "/Users/chase/Downloads/img_2879.jpg"
    hub_grey = "hubble_gray.jpg"

    compress_image(hubble, 100)
