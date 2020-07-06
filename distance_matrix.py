from sklearn.metrics import pairwise_distances
import pandas as pd

data_tables = ['mfeat-fac','mfeat-fou','mfeat-kar']

for data_table in data_tables:

    df = pd.read_csv('./data/{0}_normalized.csv'.format(data_table),delimiter=",", header=None,)

    a = pairwise_distances(df, metric='euclidean')

    df2 = pd.DataFrame(a)
    print(df2.head())

    # Save data
    df2.to_csv("./data/{0}_distance.csv".format(data_table), header=None, index=None)