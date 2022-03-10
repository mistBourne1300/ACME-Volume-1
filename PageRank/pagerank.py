# solutions.py
"""Volume 1: The Page Rank Algorithm.
<Name>
<Class>
<Date>
"""

import networkx as nx
import numpy as np
from scipy import linalg as la

# Problems 1-2
class DiGraph:
    """A class for representing directed graphs via their adjacency matrices.

    Attributes:
        (fill this out after completing DiGraph.__init__().)
    """
    # Problem 1
    def __init__(self, A, labels=None):
        """Modify A so that there are no sinks in the corresponding graph,
        then calculate Ahat. Save Ahat and the labels as attributes.

        Parameters:
            A ((n,n) ndarray): the adjacency matrix of a directed graph.
                A[i,j] is the weight of the edge from node j to node i.
            labels (list(str)): labels for the n nodes in the graph.
                If None, defaults to [0, 1, ..., n-1].
        """
        m,self.n = A.shape
        if m != self.n:
            raise ValueError("A not a square matrix")
        if not labels:
            labels = np.arange(self.n)
        if self.n != len(labels):
            raise ValueError("labels size does not match A.shape")
        for i in range(len(A[0])):
            if np.all(A[:,i]==0):
                A[:,i] = 1
        self.Ahat = A/np.sum(A, axis = 0)
        self.labels = labels

    # Problem 2
    def linsolve(self, epsilon=0.85):
        """Compute the PageRank vector using the linear system method.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.

        Returns:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        IepiAhat = np.identity(self.n) - epsilon*self.Ahat
        righthandside = np.ones(self.n)*(1-epsilon)/self.n
        p = la.solve(IepiAhat, righthandside)
        return {self.labels[i]:p[i] for i in range(len(p))}

    # Problem 2
    def eigensolve(self, epsilon=0.85):
        """Compute the PageRank vector using the eigenvalue method.
        Normalize the resulting eigenvector so its entries sum to 1.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.

        Return:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        E = np.ones((self.n,self.n))
        B = epsilon*self.Ahat + E*(1-epsilon)/self.n
        w,v = la.eig(B)
        index1 = np.argmax(w)
        p = v[:,index1]
        p = p/np.linalg.norm(p, ord = 1)
        return {self.labels[i]:p[i] for i in range(len(p))}

    # Problem 2
    def itersolve(self, epsilon=0.85, maxiter=100, tol=1e-12):
        """Compute the PageRank vector using the iterative method.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.
            maxiter (int): the maximum number of iterations to compute.
            tol (float): the convergence tolerance.

        Return:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        E = np.ones((self.n,self.n))
        B = epsilon*self.Ahat + E*(1-epsilon)/self.n
        p0 = np.ones(self.n)/self.n
        for i in range(maxiter):
            p1 = B@p0
            if np.linalg.norm(p1-p0, ord = 1) < tol:
                return {self.labels[i]:p1[i] for i in range(len(p1))}
            p0 = p1
        return {self.labels[i]:p1[i] for i in range(len(p1))}


# Problem 3
def get_ranks(d:dict):
    """Construct a sorted list of labels based on the PageRank vector.

    Parameters:
        d (dict(str -> float)): a dictionary mapping labels to PageRank values.

    Returns:
        (list) the keys of d, sorted by PageRank value from greatest to least.
    """
    values = np.array(list(d.values()))
    keys = np.array(list(d.keys()))
    sorter = np.argsort(values)
    return list(keys[sorter][::-1])


# Problem 4
def rank_websites(filename="web_stanford.txt", epsilon=0.85):
    """Read the specified file and construct a graph where node j points to
    node i if webpage j has a hyperlink to webpage i. Use the DiGraph class
    and its itersolve() method to compute the PageRank values of the webpages,
    then rank them with get_ranks(). If two webpages have the same rank,
    resolve ties by listing the webpage with the larger ID number first.

    Each line of the file has the format
        a/b/c/d/e/f...
    meaning the webpage with ID 'a' has hyperlinks to the webpages with IDs
    'b', 'c', 'd', and so on.

    Parameters:
        filename (str): the file to read from.
        epsilon (float): the damping factor, between 0 and 1.

    Returns:
        (list(str)): The ranked list of webpage IDs.
    """
    def get_unique(lines):
        unique_labels = set()
        for line in lines:
            for ID in line.strip().split('/'):
                unique_labels.add(int(ID))
        unique_labels = list(unique_labels)
        unique_labels.sort()
        return unique_labels
    
    file = open(filename)
    lines = file.readlines()
    file.close()
    labels = get_unique(lines)
    Adj = np.zeros((len(labels), len(labels)))

    for line in lines:
        line = line.strip().split('/')
        frompage = int(line[0])
        Acol = labels.index(frompage) # get the index of the column 
        for ID in line[1:]:
            ID = int(ID)
            Arow = labels.index(ID)
            Adj[Arow, Acol] = 1
      
    page_rank_dict = DiGraph(Adj,labels).itersolve(epsilon=epsilon)
    prelim = [str(i) for i in get_ranks(page_rank_dict)]
    return prelim


# Problem 5
def rank_ncaa_teams(filename, epsilon=0.85):
    """Read the specified file and construct a graph where node j points to
    node i with weight w if team j was defeated by team i in w games. Use the
    DiGraph class and its itersolve() method to compute the PageRank values of
    the teams, then rank them with get_ranks().

    Each line of the file has the format
        A,B
    meaning team A defeated team B.

    Parameters:
        filename (str): the name of the data file to read.
        epsilon (float): the damping factor, between 0 and 1.

    Returns:
        (list(str)): The ranked list of team names.
    """
    def get_unique(lines):
        unique_labels = set()
        for line in lines:
            for team in line.strip().split(','): # split across the (single) comma
                unique_labels.add(team)
        unique_labels = list(unique_labels)
        unique_labels.sort()
        return unique_labels

    file = open(filename)
    lines = file.readlines()[1:] # skip the first line, since it's a header
    file.close()
    teams = get_unique(lines)

    adj = np.zeros((len(teams), len(teams)))
    for line in lines:
        line = line.strip().split(',')
        if(len(line) != 2):
            raise ValueError(f'{line} does not have two arguments')
        col = teams.index(line[1]) # loser gets the column
        row = teams.index(line[0]) # winner gets the row
        adj[row,col] += 1
    
    page_rank_dict = DiGraph(adj,teams).itersolve(epsilon=epsilon)
    return get_ranks(page_rank_dict)
    

# Problem 6
def rank_actors(filename="top250movies.txt", epsilon=0.85):
    """Read the specified file and construct a graph where node a points to
    node b with weight w if actor a and actor b were in w movies together but
    actor b was listed first. Use NetworkX to compute the PageRank values of
    the actors, then rank them with get_ranks().

    Each line of the file has the format
        title/actor1/actor2/actor3/...
    meaning actor2 and actor3 should each have an edge pointing to actor1,
    and actor3 should have an edge pointing to actor2.
    """
    file = open(filename, encoding='utf-8')
    lines = file.readlines()
    file.close()

    hollywood = nx.DiGraph()
    for line in lines:
        line = line.strip().split('/')[1:] # strip newline, split along the '/' and the drop the movie title
        for i in range(len(line)):
            for j in range(i+1, len(line)):
                if hollywood.has_edge(line[j], line[i]):
                    hollywood[line[j]][line[i]]['weight'] += 1
                else:
                    hollywood.add_edge(line[j], line[i], weight = 1)
    ranked_actors = nx.pagerank(hollywood, alpha = epsilon)
    return get_ranks(ranked_actors)


if __name__ == "__main__":
    # A = np.array([  [0,0,0,0],
    #                 [1,0,1,0],
    #                 [1,0,0,1],
    #                 [1,0,1,0]])
    # directedgraphsolverthing = DiGraph(A, labels = ['a','b','c','d'])
    # print(get_ranks(directedgraphsolverthing.itersolve()))
    # print(rank_websites()[:10])
    # print(rank_ncaa_teams("ncaa2010.csv")[:5])
    # print(rank_actors(epsilon=0.7)[:5])
    pass

