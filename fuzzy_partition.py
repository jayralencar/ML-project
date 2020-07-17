import pandas as pd
import numpy as np
from sklearn import preprocessing



def fuzzy_partition(data, distance_matrixes, p, U = None, weights=None):
    """
    Partitioning fuzzy K-medoids clustering algorithms with relevance weight for each dissimilarity matrix estimated locally
    Parameters
    ----------
    data : 2d array, size (S, N)
        Data to be clustered.  N is the number of samples; S is the number
        of features within each sample vector.
    distance_matrixes: dissimilarity matrix
    p: number of dissimilarity matrixes
    """
    
    E = np.copy(data)
    D = np.copy(distance_matrixes)
    print(D.shape)
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

    # Randomly select K distinct prototypes
    # np.random.shuffle(E)
    # G = np.array(np.split(E,K))

    # membership degree matrix
    if U is None:
        U = np.zeros((data.shape[0], K))
        J = np.random.choice(K, data.shape[0])
        U[np.arange(data.shape[0]), J] = 1
        G = []
        for a in range(K):
            G.append(np.where(J == a)[0])
        G = np.array(G)

    else:
        # equation 11. 
        # k = 1, ... K
        # i = 1, ... n
        # total_sum = 0
        # for k in range(K):

        pass
    
    

    
    # G = np.zeros





    # Equation 9. vectors of weights calculation
    # k = 1, ... K
    # j = 1, ... p
    if weights is None:
        weights = np.zeros((K,p))

    for k in range(K):
        for j in range(p):
            arr = []
            for h in range(p): # pra fazer a produtoria
                membership_sum = 0
                for i in range(data.shape[0]):
                    dissimilarity_sum = 0
                    for e in G[k]:
                        dissimilarity_sum += D[h][i][e]
                    membership_sum += (U[i][k] ** m) * dissimilarity_sum
                arr.append(membership_sum)
            product = np.prod(np.array(arr))

            dividend = product ** 1/p # exponent
            
            divisor = 0
            for i in range(data.shape[0]):
                dissimilarity_sum = 0
                for e in G[k]:
                    dissimilarity_sum += D[j][i][e]
                divisor = (U[i][k] ** m) * dissimilarity_sum
            
            weights[k][j] = dividend/divisor
            

    print(weights)

        # print(product)

            # for ?


data_tables = ['mfeat-fac','mfeat-fou','mfeat-kar']

# load data
fact_data = pd.read_csv("./data/mfeat-fac_normalized.csv", header=None,delimiter=",")
fou_data = pd.read_csv("./data/mfeat-fou_normalized.csv", header=None,delimiter=",")
kar_data = pd.read_csv("./data/mfeat-kar_normalized.csv", header=None,delimiter=",")

# load distance matrixes
fact_dist = pd.read_csv("./data/mfeat-fac_distance.csv", header=None,delimiter=",")
fou_dist = pd.read_csv("./data/mfeat-fou_distance.csv", header=None,delimiter=",")
kar_dist = pd.read_csv("./data/mfeat-kar_distance.csv", header=None,delimiter=",")

# combine matrixes
frames = [fact_dist, fou_dist, kar_dist]
all_matrixes = pd.concat(frames)    

fuzzy_partition(fact_data, np.array([fact_dist.to_numpy()])  ,1)


