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
    # (1) Initialization (pag. 13)

    # Fix K (the number of clusters)
    K = 10
    # fix m
    m = 1.6
    # fix s
    s = 1
    # fix T (an interation limit)
    T = 2
    # fix epsilon
    epsilon = 10 ** -10
    # set the cardinality 1 =< q << n ?????????
    q = 10
    # set t = 0
    t = 0

    #
    # Set weights
    weights = np.zeros((K,p))
    for k in range(K):
        weights[k] = [1]*p
    

    # Randomly select K distinct prototypes
    G = np.zeros((K,q), dtype=int)
    J = np.random.choice(K, data.shape[0])
    for k in range(K):
        G[k] = np.random.choice(np.where(J == k)[0], q)

    # membership degree matrix - empty
    U = np.zeros((data.shape[0], K))

    while t < T:
        t += 1
        # equation 11. 
        # k = 1, ... K
        # i = 1, ... n
        # total_sum = 0
        print("membership degree matrix")
        for i in range(data.shape[0]):
            for k in range(K):
                total_sum = 0
                for h in range(K):
                    sum_weigths = 0
                    for j in range(p):
                        
                        dissimilarity_sum = 0
                        for e in G[k]:
                            dissimilarity_sum += D[j][i][e]

                        sum_weigths += (weights[k][j]**s) * dissimilarity_sum
                    
                    dividend = sum_weigths

                    sum_weigths2 = 0
                    for j in range(p):
                        dissimilarity_sum2 = 0
                        for e in G[h]:
                            dissimilarity_sum2 += D[j][i][e]
                        sum_weigths2 += (weights[h][j]**s) * dissimilarity_sum2
                    
                    divisor = sum_weigths2 

                    res = (dividend/divisor)**(1/(m-1))

                    total_sum += res
                # print()
                U[i][k] = total_sum ** -1
        print("membership")
        print(U)
        

        # prototypes
        # Proposition 2.3
        print("Fiding prototypes")
        for k in range(K):
            G_asteristic = []
            sums = []
            for h in range(data.shape[0]):
                sum_ = 0
                for i in range(data.shape[0]):
                    dissimilarity_sum = 0
                    for j in range(p):
                        # print(D[j][i][h])
                        dissimilarity_sum += (weights[k][j]**s) * D[j][i][h]
                    # print("siss", dissimilarity_sum)
                    sum_ += (U[i][k]**m)*dissimilarity_sum
                    # print("summ",sum_)
                sums.append(sum_)
            # l = np.argmin()
            # print("SUMS", sums)
            G_asteristic = np.array(sums).argsort()[:q]
                        # for e in G[k]:
                            # dissimilarity_sum += D[j][i][e]
            G[k] = G_asteristic
        print(G)


        # Equation 9. vectors of weights calculation
        # k = 1, ... K
        # j = 1, ... p
        print("vectors of weights calculation")
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

                dividend = product ** (1/p) # exponent
                
                divisor = 0
                for i in range(data.shape[0]):
                    dissimilarity_sum = 0
                    for e in G[k]:
                        dissimilarity_sum += D[j][i][e]

                    divisor += (U[i][k] ** m) * dissimilarity_sum
                
                weights[k][j] = dividend/divisor
                
        print("weights")
        print(weights)


        

        

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

# fuzzy_partition(np.array([kar_data.to_numpy()]), np.array([kar_dist.to_numpy()])  ,1)
fuzzy_partition(fact_data, np.array([fact_dist.to_numpy()])  ,1)

