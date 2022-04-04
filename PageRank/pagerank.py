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
        # check to see if A is square
        if m != self.n:
            raise ValueError("A not a square matrix")
        
        # create labels if there are none
        if not labels:
            labels = np.arange(self.n)
        
        # throw error if labels are not the right size
        if self.n != len(labels):
            raise ValueError("labels size does not match A.shape")
        
        # rows of zeros become rows of ones
        for i in range(len(A[0])):
            if np.all(A[:,i]==0):
                A[:,i] = 1
        
        # create Ahat and save as self objects
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
        # set up, create the linear system to solve
        IepiAhat = np.identity(self.n) - epsilon*self.Ahat
        righthandside = np.ones(self.n)*(1-epsilon)/self.n
        
        # get the solution by brute force
        p = la.solve(IepiAhat, righthandside)

        # return the dictionary with the labels
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
        p = p/np.sum(p)
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
        ones = np.ones(self.n)
        p0 = np.ones(self.n)/self.n
        for i in range(maxiter):
            p1 = epsilon * self.Ahat@p0 + (1-epsilon) * ones / self.n
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
    # get values and keys as lists
    l = [(-val,key) for key,val in zip(d.keys(), d.values())]

    # sort the dictionary and return
    l.sort()
    return [i[1] for i in l]

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

    file = open(filename)
    data = file.read().strip()
    file.close()

    # get the unique labels from the helper function
    labels = sorted(set(data.replace('\n', '/').split('/')))
    Adj = np.zeros((len(labels), len(labels)))

    indices = {site:i for i,site in enumerate(labels)}

    # loop through each line
    for line in data.split('\n'):
        line = line.strip().split('/') # strip and split the line across the '/'
        frompage = line[0]
        Acol = indices[frompage]
        # loop through the rest of the line
        for ID in line[1:]:
            Arow = indices[ID]
            Adj[Arow, Acol] += 1
    
    # get dictionary from the DiGraph class
    page_rank_dict = DiGraph(Adj,labels).itersolve(epsilon=epsilon)

    # return the sorted list for the ranked pages
    return get_ranks(page_rank_dict)

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
    # get unique teams list
    def get_unique(lines):
        unique_labels = set()
        for line in lines:
            for team in line.strip().split(','): # split across the (single) comma
                unique_labels.add(team)
        unique_labels = list(unique_labels)
        unique_labels.sort()
        return unique_labels

    # read file
    file = open(filename)
    lines = file.readlines()[1:] # skip the first line, since it's a header
    file.close()

    # get unique teams using helper function
    teams = get_unique(lines)

    # create empty adj matrix
    adjective = np.zeros((len(teams), len(teams)))

    # fill adj matrix with data
    for line in lines:
        line = line.strip().split(',')
        if(len(line) != 2):
            raise ValueError(f'{line} does not have two arguments')
        col = teams.index(line[1]) # loser gets the column
        row = teams.index(line[0]) # winner gets the row
        adjective[row,col] += 1
    
    # get the dictionary from the DiGraph class and return
    page_rank_dict = DiGraph(adjective,teams).itersolve(epsilon=epsilon)
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
    # read in and save file lines
    file = open(filename, encoding='utf-8')
    lines = file.readlines()
    file.close()

    # create new nx directed graph
    hollywood = nx.DiGraph()
    for line in lines:
        line = line.strip().split('/')[1:] # strip newline, split along the '/' and the drop the movie title
        # for each higher-billed actor
        for i in range(len(line)):
            # for each lower-billed actor
            for j in range(i+1, len(line)):
                # if there is already and edge, up the weight by one
                if hollywood.has_edge(line[j], line[i]):
                    hollywood[line[j]][line[i]]['weight'] += 1
                # otherwise, create an edge
                else:
                    hollywood.add_edge(line[j], line[i], weight = 1)
    # get the highest ranked actors from the pagerank and return
    ranked_actors = nx.pagerank(hollywood, alpha = epsilon)
    return get_ranks(ranked_actors)


if __name__ == "__main__":
    # A = np.array([  [0,0,0,0],
    #                 [1,0,1,0],
    #                 [1,0,0,1],
    #                 [1,0,1,0]])
    # directedgraphsolverthing = DiGraph(A, labels = ['a','b','c','d'])
    # print(directedgraphsolverthing.eigensolve())
    # print(directedgraphsolverthing.itersolve())
    # print(get_ranks(directedgraphsolverthing.itersolve()))
    print(rank_websites(epsilon=0.75)[:21])
    # print(rank_ncaa_teams("ncaa2010.csv")[:5])
    print(rank_actors(epsilon=0.7)[:5])
    pass

