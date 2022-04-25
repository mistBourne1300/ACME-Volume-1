# drazin.py
"""Volume 1: The Drazin Inverse.
<Name>
<Class>
<Date>
"""

import numpy as np
from scipy import linalg as la


# Helper function for problems 1 and 2.
def index(A, tol=1e-5):
    """Compute the index of the matrix A.

    Parameters:
        A ((n,n) ndarray): An nxn matrix.

    Returns:
        k (int): The index of A.
    """

    # test for non-singularity
    if not np.isclose(la.det(A), 0):
        return 0

    n = len(A)
    k = 1
    Ak = A.copy()
    while k <= n:
        r1 = np.linalg.matrix_rank(Ak)
        r2 = np.linalg.matrix_rank(np.dot(A,Ak))
        if r1 == r2:
            return k
        Ak = np.dot(A,Ak)
        k += 1

    return k

# Problem 1
def is_drazin(A, Ad, k):
    """Verify that a matrix Ad is the Drazin inverse of A.

    Parameters:
        A ((n,n) ndarray): An nxn matrix.
        Ad ((n,n) ndarray): A candidate for the Drazin inverse of A.
        k (int): The index of A.

    Returns:
        (bool) True of Ad is the Drazin inverse of A, False otherwise.
    """
    # check A@Ad == Ad@A
    if not np.allclose(A@Ad, Ad@A):
        return False
    
    # check Ad@A@Ad == Ad
    if not np.allclose(Ad@A@Ad,Ad):
        return False
    
    # check (A**k+1)@Ad == A**k
    Ak = np.linalg.matrix_power(A,k)
    if not np.allclose(A@Ak@Ad,Ak):
        return False
    
    return True

# Problem 2
def drazin_inverse(A, tol=1e-4):
    """Compute the Drazin inverse of A.

    Parameters:
        A ((n,n) ndarray): An nxn matrix.

    Returns:
       ((n,n) ndarray) The Drazin inverse of A.
    """
    # get shape
    n = A.shape[0]
    # define "zero" functions
    f = lambda x: abs(x) > tol
    g = lambda x: abs(x) <= tol

    # get T1,Q1,k1 and T2,Q2,k2
    T1,Q1,k1 = la.schur(A,sort = f)
    T2,Q2,k2 = la.schur(A,sort = g)

    # stack Q1 and Q2 (up to the relevant index) to create U
    U = np.hstack((Q1[:,:k1],Q2[:,:n-k1]))
    # get U inverse
    U_1 = la.inv(U)
    # create V and initialize Z
    V = U_1@A@U
    Z = np.zeros((n,n))

    # if necessary, remake Z to be int inverse of V
    if k1 != 0:
        Z[:k1,:k1] = la.inv(V[:k1,:k1])
    return U@Z@U_1

# Problem 3
def effective_resistance(A):
    """Compute the effective resistance for each node in a graph.

    Parameters:
        A ((n,n) ndarray): The adjacency matrix of an undirected graph.

    Returns:
        ((n,n) ndarray) The matrix where the ijth entry is the effective
        resistance from node i to node j.
    """
    # create lapplacian
    laplacian = np.diag(A.sum(axis = 1)) - A
    # initialize R and create I
    R = np.zeros(A.shape)
    I = np.eye(A.shape[0])

    # for each column, copy the laplacian and replace the column with the identity
    # Get the Drazin inverse, and push the diagonals to the column of R
    for j in range(A.shape[0]):
        Lj = laplacian.copy()
        Lj[j,:] = I[j,:]
        Ljd = drazin_inverse(Lj)
        R[:,j] = np.diag(Ljd)
    return R - I

# Problems 4 and 5
class LinkPredictor:
    """Predict links between nodes of a network."""

    def __init__(self, filename='social_network.csv'):
        """Create the effective resistance matrix by constructing
        an adjacency matrix.

        Parameters:
            filename (str): The name of a file containing graph data.
        """
        # go through the file, pushing each name into the set of names 
        names = set()
        with open(filename) as file:
            for line in file.readlines():
                for name in line.strip().split(','):
                    names.add(name)
        # make names a list and sort it
        self.names = list(names)
        self.names.sort()
        
        # initialize Adjacency as zeros
        self.Adj = np.zeros((len(names),len(names)))

        # go through the file again, adding adjacencies between the names
        with open(filename) as file:
            for line in file.readlines():
                name1,name2 = line.strip().split(',')
                i,j = self.names.index(name1), self.names.index(name2)
                self.Adj[i,j] += 1
                self.Adj[j,i] += 1
        
        # create the resistance matrix
        self.ER = effective_resistance(self.Adj)

    def predict_link(self, node=None):
        """Predict the next link, either for the whole graph or for a
        particular node.

        Parameters:
            node (str): The name of a node in the network.

        Returns:
            node1, node2 (str): The names of the next nodes to be linked.
                Returned if node is None.
            node1 (str): The name of the next node to be linked to 'node'.
                Returned if node is not None.

        Raises:
            ValueError: If node is not in the graph.
        """
        # copy ER and the Adjacency matrix
        ER = self.ER.copy()
        Adj = self.Adj.copy()

        # create a mask where there are connections in the Adjacency matrix
        # then make those entries zero in ER
        mask = Adj!=0
        ER[mask] = 0

        # if node was not passed in, get the overall closest non-linked people
        if not node:
            # get the nonzero minimum
            minimum = np.min(ER[np.nonzero(ER)])
            # get the index of the nonzero minimum and return
            row,col = np.where(ER == minimum)
            return self.names[row[0]],self.names[col[0]]
        
        else:
            # get the appropriate row corresponding to node
            i = self.names.index(node)
            # get the nonzero minimum along the correct row
            minimum = np.min(ER[i,np.nonzero(ER[i,:])])
            # get index onf minimum
            ind = np.where(ER[i,:] == minimum)
            return self.names[ind[0][0]]

    def add_link(self, node1, node2):
        """Add a link to the graph between node 1 and node 2 by updating the
        adjacency matrix and the effective resistance matrix.

        Parameters:
            node1 (str): The name of a node in the network.
            node2 (str): The name of a node in the network.

        Raises:
            ValueError: If either node1 or node2 is not in the graph.
        """
        # get row and column indices
        row = self.names.index(node1)
        col = self.names.index(node2)
        # create adjacencies between node1 and node2
        self.Adj[row,col] = 1
        self.Adj[col,row] = 1
        # recalculate resistance matrix
        self.ER = effective_resistance(self.Adj)


if __name__ == "__main__":
    A = np.array([  [1,3,0,0],
                    [0,1,3,0],
                    [0,0,1,3],
                    [0,0,0,0]])
    Ad = np.array([ [1,-3,9,81],
                    [0,1,-3,-18],
                    [0,0,1,3],
                    [0,0,0,0]])
    k_A = 1
    
    B = np.array([  [1,1,3],
                    [5,2,6],
                    [-2,-1,-3]])
    Bd = np.zeros((3,3))
    k_B = 3

    print("PROBLEM 1:\n")
    print(f'A: {is_drazin(A,Ad,k_A)}')
    print(f'B: {is_drazin(B,Bd,k_B)}')

    print("\n\nPROBLEM 2:\n")
    print(f'A: {is_drazin(A,drazin_inverse(A),k_A)}')
    print(f'B: {is_drazin(B,drazin_inverse(B),k_B)}')

    print("\n\nPROBLEM 3:\n")
    graph1 = np.array([ [0,1,0,0],
                        [1,0,1,0],
                        [0,1,0,1],
                        [0,0,1,0]])
    print("graph 1:")
    print(effective_resistance(graph1))

    print("\n\nPROBLEM 4:\n")
    Zelda = LinkPredictor()
    print(f'overall: {Zelda.predict_link()}')
    print(f'Melanie: {Zelda.predict_link("Melanie")}')
    print(f'Alan: {Zelda.predict_link("Alan")}')
    Zelda.add_link("Alan", "Sonia")
    print("added link between Alan and Sonia")
    print(f'Alan: {Zelda.predict_link("Alan")}')
    Zelda.add_link("Alan", "Piers")
    print("added link between Alan and Piers")
    print(f'Alan: {Zelda.predict_link("Alan")}')
