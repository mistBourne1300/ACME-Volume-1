# image_segmentation.py
"""Volume 1: Image Segmentation.
<Name>
<Class>
<Date>
"""

import numpy as np
from numpy.linalg import eig
from scipy import linalg as la
from scipy import sparse
import scipy.sparse.linalg as spla
import imageio
from matplotlib import pyplot as plt
import os

# Problem 1
def laplacian(A):
    """Compute the Laplacian matrix of the graph G that has adjacency matrix A.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.

    Returns:
        L ((N,N) ndarray): The Laplacian matrix of G.
    """
    D = np.diag(A.sum(axis = 1))
    return D-A


# Problem 2
def connectivity(A, tol=1e-8):
    """Compute the number of connected components in the graph G and its
    algebraic connectivity, given the adjacency matrix A of G.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.
        tol (float): Eigenvalues that are less than this tolerance are
            considered zero.

    Returns:
        (int): The number of connected components in G.
        (float): the algebraic connectivity of G.
    """
    # does not work yet
    L = laplacian(A)
    eigs = la.eigvals(L)
    eigs = [np.real(v) for v in eigs]
    eigs = np.sort(eigs)
    for i in range(len(eigs)):
        if(eigs[i] > tol):
            return i, eigs[i]


# Helper function for problem 4.
def get_neighbors(index, radius, height, width):
    """Calculate the flattened indices of the pixels that are within the given
    distance of a central pixel, and their distances from the central pixel.

    Parameters:
        index (int): The index of a central pixel in a flattened image array
            with original shape (radius, height).
        radius (float): Radius of the neighborhood around the central pixel.
        height (int): The height of the original image in pixels.
        width (int): The width of the original image in pixels.

    Returns:
        (1-D ndarray): the indices of the pixels that are within the specified
            radius of the central pixel, with respect to the flattened image.
        (1-D ndarray): the euclidean distances from the neighborhood pixels to
            the central pixel.
    """
    # Calculate the original 2-D coordinates of the central pixel.
    row, col = index // width, index % width

    # Get a grid of possible candidates that are close to the central pixel.
    r = int(radius)
    x = np.arange(max(col - r, 0), min(col + r + 1, width))
    y = np.arange(max(row - r, 0), min(row + r + 1, height))
    X, Y = np.meshgrid(x, y)

    # Determine which candidates are within the given radius of the pixel.
    R = np.sqrt(((X - col)**2 + (Y - row)**2))
    mask = R < radius
    return (X[mask] + Y[mask]*width).astype(int), R[mask]


# Problems 3-6
class ImageSegmenter:
    """Class for storing and segmenting images."""

    # Problem 3
    def __init__(self, filename):
        """Read the image file. Store its brightness values as a flat array."""
        self.image = imageio.imread(filename) / 255
        if len(self.image.shape) == 3:
            self.greyscale = self.image.mean(axis=2)
        else:
            self.greyscale = self.image
        self.unravelled = np.ravel(self.greyscale)
        

    # Problem 3
    def show_original(self):
        """Display the original image."""
        if len(self.image.shape) == 3:
            plt.imshow(self.image)
        else:
            plt.imshow(self.image, cmap = "gray")
        plt.axis("off")
        plt.show()

    # Problem 4
    def adjacency(self, r=5., sigma_B2=.02, sigma_X2=3.):
        """Compute the Adjacency and Degree matrices for the image graph."""
        h, w = self.greyscale.shape
        A = sparse.lil_matrix((len(self.unravelled), len(self.unravelled)))
        D=[]
        for i in range(A.shape[0]):
            neighbors, distances = get_neighbors(i, r, h, w)
            weights = []
            for j in range(len(neighbors)):
                exleft = np.abs(self.unravelled[i] - self.unravelled[neighbors[j]])/sigma_B2
                exright = distances[j]/sigma_X2
                wij = np.exp(- exleft - exright)
                weights.append(wij)
            A[i, np.array(neighbors)] = weights
            D.append(sum(weights))
        A = A.tocsc()
        D = np.array(D)
        return A, D
        raise NotImplementedError("Problem 4 Incomplete")

    # Problem 5
    def cut(self, A, D):
        """Compute the boolean mask that segments the image."""
        h,w = self.greyscale.shape
        L = sparse.csgraph.laplacian(A)
        over_sqrtD  = sparse.diags([1/np.sqrt(d) for d in D])
        # returns the two smallest eigenvalues and eigenvectors
        eigvals, eigvects = spla.eigsh(over_sqrtD @ L @ over_sqrtD, which = "SM", k=2)
        max_index = np.argmax(eigvals)
        second_smallest_eigvect = np.array(eigvects[:,max_index]).reshape((h,w))
        return second_smallest_eigvect > 0 

        raise NotImplementedError("Problem 5 Incomplete")

    # Problem 6
    def segment(self, r=5., sigma_B=.02, sigma_X=3.):
        """Display the original image and its segments."""
        A, D = self.adjacency(r,sigma_B, sigma_X)
        cut = self.cut(A,D)
        if len(self.image.shape) == 3:
            # we have a color image
            cutlist = [cut for c in range(self.image.shape[-1])]
            cut3 = np.stack(cutlist, axis = 2)
            
            plt.subplot(131)
            plt.imshow(self.image)
            plt.axis('off')

            plt.subplot(132)
            plt.imshow(self.image * cut3)
            plt.axis('off')

            plt.subplot(133)
            plt.imshow(self.image * ~cut3)
            plt.axis('off')
        else:
            plt.subplot(131)
            plt.imshow(self.image, cmap = 'gray')
            plt.axis('off')

            plt.subplot(132)
            plt.imshow(self.image * cut, cmap = 'gray')
            plt.axis('off')

            plt.subplot(133)
            plt.imshow(self.image * ~cut, cmap = 'gray')
            plt.axis('off')

        plt.show()


if __name__ == '__main__':
    os.chdir("/Users/chase/Desktop/Math345Volume1/byu_vol1/ImageSegmentation")
    ImageSegmenter("dream_gray.png").segment()
    ImageSegmenter("dream.png").segment()
    ImageSegmenter("blue_heart.png").segment()

if __name__ == "__main__":
    # A = np.array([  [0,1,0,0,1,1],
    #                 [1,0,1,0,1,0],
    #                 [0,1,0,1,0,0],
    #                 [0,0,1,0,1,1],
    #                 [1,1,0,1,0,0],
    #                 [1,0,0,1,0,0]])
    # B = np.array([  [0,3,0,0,0,0],
    #                 [3,0,0,0,0,0],
    #                 [0,0,0,1,0,0],
    #                 [0,0,1,0,2,.5],
    #                 [0,0,0,2,0,1],
    #                 [0,0,0,.5,1,0]])
    # print(laplacian(A))
    # L = laplacian(A)
    # eigs = la.eigvals(L)
    # eigs = [np.real(e) for e in eigs]
    # eigs = np.sort(eigs)
    # print(eigs)
    # print(connectivity(B))
    os.chdir("/Users/chase/Desktop/Math345Volume1/byu_vol1/ImageSegmentation")

    im1 = "/Users/chase/Downloads/img_0124.jpg"
    im2 = "dream_gray.png"
    im3 = "blue_heart.png"
    im4 = "/Users/chase/Downloads/istockphoto-1226241649-170667a.jpg"
    im5 = "dream.png"
    chimera = ImageSegmenter(im2)
    # chimera.show_original()
    A, D = chimera.adjacency()
    # A = sparse.csr_matrix.toarray(A)
    # print(f'A shape: {A.shape}, data type: {A.dtype}')
    # print(f'D: {D}, data type: {D.dtype}')
    # blueH_A = np.load("HeartMatrixA.npz").files
    # bluH_D = np.load("HeartMatrixD.npy")
    # print(f'A should be: {blueH_A}')
    # print(f'D should be: {bluH_D}')
    # print(f'mask: {chimera.cut(A,D)}')
    chimera.segment()
    pass