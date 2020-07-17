import pandas as pd
import numpy as np
from sklearn import preprocessing



def fuzzy_partition(data, distance_matrix, p):
    """
    Partitioning fuzzy K-medoids clustering algorithms with relevance weight for each dissimilarity matrix estimated locally
    Parameters
    ----------
    data : 2d array, size (S, N)
        Data to be clustered.  N is the number of samples; S is the number
        of features within each sample vector.
    distance_matrix: dissimilarity matrix
    p: number of dissimilarity matrixes
    """
    
    E = np.copy(data)

    # (1) Initialization (pag. 6)

    # Fix K (the number of clusters)
    K = 10
    # fix m
    m = 1.6
    # fix T (an interation limit)
    T = 150
    # fix epsilon
    epsilon = 10 ** -10
    # set the cardinality 1 =< q << n ?????????
    q = 1 
    # set t = 0
    t = 0

    # p is the dissimilarity matrix length
    p = 3

    # Randomly select K distinct prototypes
    np.random.shuffle(E)
    G = np.array(np.split(E,K))

    # equation 11. membership degree
    # k = 1, ... K
    # i = 1, ... n

    


    # Equation 9. vectors of weights calculation
    # k = 1, ... K
    # j = 1, ... p
    
    for k in range(K):
        for j in range(p):
            pass


data_tables = ['mfeat-fac','mfeat-fou','mfeat-kar']

# load data
fact_data = pd.read_csv("./data/mfeat-fac_normalized.csv", header=None)
fou_data = pd.read_csv("./data/mfeat-fou_normalized.csv", header=None)
kar_data = pd.read_csv("./data/mfeat-kar_normalized.csv", header=None)

# load distance matrixes
fact_dist = pd.read_csv("./data/mfeat-fac_distance.csv", header=None)
fou_dist = pd.read_csv("./data/mfeat-fou_distance.csv", header=None)
kar_dist = pd.read_csv("./data/mfeat-kar_distance.csv", header=None)

# combine matrixes
frames = [fact_dist, fou_dist, kar_dist]
all_matrixes = pd.concat(frames)    

fuzzy_partition(fact_data, fact_dist,1)


