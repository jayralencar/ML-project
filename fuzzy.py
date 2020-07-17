import pandas as pd
import numpy as np
from sklearn import preprocessing

data_tables = ['mfeat-fac','mfeat-fou','mfeat-kar']

fact = pd.read_csv("./data/mfeat-fac_distance.csv", header=None)
fou = pd.read_csv("./data/mfeat-fou_distance.csv", header=None)
kar = pd.read_csv("./data/mfeat-kar_distance.csv", header=None)

frames = [fact, fou, kar]

all_matrixes = pd.concat(frames)

# a = fact.to_numpy()
# print(a.shape)
def normalize_columns(data):
    df = pd.DataFrame(data)
    std_scale = preprocessing.StandardScaler().fit(df)
    x_train_norm = std_scale.transform(df)

    df2 = pd.DataFrame(x_train_norm)

    return df2.to_numpy()


def _fp_coeff(u):
    """
    Fuzzy partition coefficient `fpc` relative to fuzzy c-partitioned
    matrix `u`. Measures 'fuzziness' in partitioned clustering.
    Parameters
    ----------
    u : 2d array (C, N)
        Fuzzy c-partitioned matrix; N = number of data points and C = number
        of clusters.
    Returns
    -------
    fpc : float
        Fuzzy partition coefficient.
    """
    n = u.shape[1]

    return np.trace(u.dot(u.T)) / float(n)

def fuzzy_partition(data,c,maxiter, error, init=None, seed=None):
    """
    Partitioning fuzzy K-medoids clustering algorithms with relevance weight for each dissimilarity matrix estimated locally
    Parameters
    ----------
    data : 2d array, size (S, N)
        Data to be clustered.  N is the number of samples; S is the number
        of features within each sample vector.
    c : int
        Desired number of clusters or classes.
    maxiter : int
        Maximum number of iterations allowed.
    init : 2d array, size (S, N)
        Initial fuzzy c-partitioned matrix. If none provided, algorithm is
        randomly initialized.
    seed : int
        If provided, sets random seed of init. No effect if init is
        provided. Mainly for debug/testing purposes.
    """
     # Setup u0 (Initialization)
    if init is None:
        if seed is not None:
            np.random.seed(seed=seed)
        n = data.shape[1]
        u0 = np.random.rand(c, n)
        u0 = normalize_columns(u0)
        init = u0.copy()
    u0 = init
    
    u = np.fmax(u0, np.finfo(np.float64).eps)

    # Initialize loop parameters
    jm = np.zeros(0)
    p = 0

    # while p < maxiter - 1:
    #     u2 = u.copy()
    #     p += 1
    #     [cntr, u, Jjm, d] = fuzzy_partition(data, u2, c,error)
    #     jm = np.hstack((jm, Jjm))
    #     p += 1

    #     # Stopping rule
    #     if np.linalg.norm(u - u2) < error:
    #         break

    # # Final calculations
    # error = np.linalg.norm(u - u2)
    # fpc = _fp_coeff(u)

    # return cntr, u, u0, d, jm, p, fpc

    

print(fuzzy_partition(fact.to_numpy(),10,100,0.005))