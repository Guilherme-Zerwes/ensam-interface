import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import os

# def reduce_order(target):
#     df = pd.read_csv('data/processed_dataset.csv') if os.path.exists('data/processed_dataset.csv') else pd.read_csv('data/dataset.csv')
#     df.drop(columns='Unnamed: 0', inplace=True)
#     X = df.drop(columns=target)
#     y = df[target]

#     pca = PCA().fit(X=X, y=y)
#     component_counter = 0
#     var_explained = 0
#     threshold = 0.8

#     for i in range(len(pca.components_)):
#         var_explained += pca.explained_variance_ratio_[i]
#         component_counter += 1
#         if var_explained >= threshold:
#             break
    
#     reduced_data = PCA(n_components=component_counter).fit_transform(X=X, y=y)

#     df = pd.DataFrame(reduced_data, columns=list(range(component_counter)))
#     df[target] = y
#     df.to_csv('data/processed_dataset.csv')
#     print(df.head())
#     return reduced_data

def reduce_order(target, data):
    '''Proper Orthogonal Decomposition'''
    df = data
    df.drop(columns='Unnamed: 0', inplace=True)
    X = df.drop(columns=target)
    y = df[target]

    mean = np.mean(np.transpose(X))
    X_new = np.transpose(X) - mean

    U, S, VT = np.linalg.svd(X_new, full_matrices=False)

    threshold = 0.9
    rank = np.argwhere(np.cumsum(S/np.sum(S)) > threshold)[0][0] + 1
    S = np.diag(S[0:rank])
    fluctuation = np.dot(U[0:rank,0:rank], np.dot(S, VT[0:rank,:]))
    reduced_data = mean + fluctuation

    df = pd.DataFrame(np.transpose(reduced_data), columns=list(range(rank)))
    df[target] = y
    print('Order recuded from', X.shape, 'to', np.transpose(reduced_data).shape)
    # df.to_csv('data/processed_dataset.csv')
    return df
