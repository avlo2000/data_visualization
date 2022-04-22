import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
from keras.utils import to_categorical


def do_pca(df):
    df_PCA = df.drop(columns=['Gender', 'CLASS'])
    np_pca = df_PCA.to_numpy()

    scaler = preprocessing.MinMaxScaler()
    x_scaled = scaler.fit_transform(np_pca)

    pca = PCA(n_components=len(df_PCA.columns))
    pca.fit(x_scaled)
    print(pca.explained_variance_ratio_)
    print(pca.singular_values_)

    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.show()


def prepare_data(df: pd.DataFrame):
    y = pd.Categorical(df['CLASS']).codes

    df = df.drop(columns=['CLASS'])
    df['Gender'] = pd.Categorical(df['Gender']).codes
    x = df.to_numpy()


def main():
    df = pd.read_csv('data/Dataset of Diabetes .csv')

    print(df.columns)
    df = df.drop(columns=['ID'])
    print(df.head(10))

    do_pca(df)


if __name__ == '__main__':
    main()

