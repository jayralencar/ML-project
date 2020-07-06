from sklearn import preprocessing
import pandas as pd

data_tables = ['mfeat-fac','mfeat-fou','mfeat-kar']

for data_table in data_tables:

    # read data
    df = pd.read_csv('./data/{0}'.format(data_table),delimiter=r"\s+", header=None)

    # show head
    print(df.head())

    # I will be normalizing features by removing the mean and scaling it to unit variance.
    std_scale = preprocessing.StandardScaler().fit(df)
    x_train_norm = std_scale.transform(df)

    df2 = pd.DataFrame(x_train_norm)
    print(df2.head())

    # Save data
    df2.to_csv("./data/{0}_normalized.csv".format(data_table), header=None, index=None)
