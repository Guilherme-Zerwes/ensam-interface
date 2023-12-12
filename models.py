import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split

import visualization


def train_reg_model(algo, target, hyper, data):
    df = data
    X = df.drop(columns=target)
    y = df[target]
    x_train, x_test, y_train, y_test = train_test_split(np.asarray(X), np.asarray(y))

    if algo == 'Linear Regression':
        from sklearn.linear_model import LinearRegression

        model = LinearRegression()
        model.fit(x_train,y_train)
        
        metrics = visualization.reg_vis(algo, model, x_test, y_test, [X,y])

        #Code for the lin reg plot
        x_vis = x_train[:, np.argmax(np.abs(model.coef_))]
        y_vis = LinearRegression().fit(x_vis.reshape(-1, 1), y_train).predict(x_vis.reshape(-1, 1))
        
        fig_vis,ax_vis = plt.subplots()
        ax_vis.set_title('Linear Regression Plot')
        ax_vis.set_xlabel(f'{list(df.keys())[np.argmax(np.abs(model.coef_))]}')
        ax_vis.set_ylabel('Values')
        ax_vis.grid()
        fig_vis.tight_layout()
        ax_vis.scatter(x_vis, y_train, label='Data points')
        ax_vis.plot(x_vis, y_vis, 'r--', label='Regression model')
        metrics.append(fig_vis)

    elif algo == 'Polynomial Regression':
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression

        degree = int(hyper[0])
        pol_fits = PolynomialFeatures(degree=degree)
        x_original = x_train.copy()
        x_train = pol_fits.fit_transform(x_train)
        x_test =  pol_fits.fit_transform(x_test)

        model = LinearRegression()
        model.fit(x_train,y_train)
        
        metrics = visualization.reg_vis(algo, model, x_test, y_test, [X,y])

        #Code for the poly reg plot
        x_vis = x_original[:, 2]
        x_vis = np.sort(x_vis)
        y_train = y_train[np.argsort(x_vis)]

        x_vis = pol_fits.fit_transform(x_vis.reshape(-1, 1))
        
        y_vis = LinearRegression().fit(x_vis, y_train).predict(x_vis)
        
        fig_vis,ax_vis = plt.subplots()
        ax_vis.set_title('Polynomial Regression Plot')
        # ax_vis.set_xlabel(f'{list(df.keys())[np.argmax(np.abs(model.coef_))]}')
        ax_vis.set_ylabel('Values')
        ax_vis.grid()
        fig_vis.tight_layout()

        ax_vis.scatter(x_vis[:,1], y_train, label='Data points')
        ax_vis.plot(x_vis[:,1], y_vis, 'r--', label='Regression model')
        metrics.append(fig_vis)

    elif algo == 'Random Forests Regressor':
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=int(hyper[0]), max_depth=int(hyper[1]))
        model.fit(x_train,y_train)
        
        metrics = visualization.reg_vis(algo, model, x_test, y_test, [X,y])


    elif algo == 'Support Vector Machines Regression':
        from sklearn.svm import SVR

        model = SVR(kernel=hyper[0], C=hyper[1])
        model.fit(x_train,y_train)
        
        metrics = visualization.reg_vis(algo, model, x_test, y_test, [X,y])
        

    elif algo == 'K-Nearest-Neighbors Regressor':
        from sklearn.neighbors import KNeighborsRegressor

        model = KNeighborsRegressor(n_neighbors=int(hyper[0]))
        model.fit(x_train,y_train)
        
        metrics = visualization.reg_vis(algo, model, x_test, y_test, [X,y])
        

    elif algo == 'Artificial Neural Networks Regressor':
        from sklearn.neural_network import MLPRegressor
        hidden = np.ones((int(hyper[0])), dtype=int)*int(hyper[1])

        model = MLPRegressor(hidden_layer_sizes=hidden)
        model.fit(x_train,y_train)
        
        metrics = visualization.reg_vis(algo, model, x_test, y_test, [X,y])
        

    pickle.dump(model, open('./data/model.pickle', 'wb'))
    return metrics

####################################

def train_class_model(algo, target, hyper, data):
    df = data
    X = df.drop(columns=target)
    y = df[target]
    x_train, x_test, y_train, y_test = train_test_split(np.asarray(X), np.asarray(y))

    if algo == 'Decision Tree Classifier':
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(max_depth=int(hyper[0]))
        model.fit(x_train,y_train)
        
        metrics = visualization.class_vis(algo, model, x_test, y_test, [X,y])

    elif algo == 'Random Forests Classifier':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=int(hyper[0]), max_depth=int(hyper[1]))
        model.fit(x_train,y_train)
        
        metrics = visualization.class_vis(algo, model, x_test, y_test, [X,y])

    elif algo == 'Support Vector Machine Classifier':
        from sklearn.svm import SVC
        model = SVC(kernel=hyper[0], C=hyper[1])
        model.fit(x_train,y_train)
        
        metrics = visualization.class_vis(algo, model, x_test, y_test, [X,y])

    elif algo == 'K-Nearest-Neighbors Classifier':
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(n_neighbors=int(hyper[0]))
        model.fit(x_train,y_train)
        
        metrics = visualization.class_vis(algo, model, x_test, y_test, [X,y])

    elif algo == 'Artificial Neural Networks Classifier':
        from sklearn.neural_network import MLPClassifier
        hidden = np.ones((int(hyper[0])), dtype=int)*int(hyper[1])

        model = MLPClassifier(hidden_layer_sizes=hidden)
        model.fit(x_train,y_train)
        metrics = visualization.class_vis(algo, model, x_test, y_test, [X,y])
        
    pickle.dump(model, open('./data/model.pickle', 'wb'))
    return metrics

####################################

def train_desc_model(algo, hyper, data, no = None):
    df = data
    
    if no != None:
        df = df.drop(columns=no)

    if algo == 'Principal Component Analysis':
        from sklearn.decomposition import PCA
        model = PCA(n_components=int(hyper[0]))
        reduced_data = model.fit_transform(np.asarray(df))

        metrics = visualization.desc_vis(algo, model, reduced_data, df)

    elif algo == 'Kernel Principal Component Analysis':
        from sklearn.decomposition import KernelPCA
        model = KernelPCA(n_components=int(hyper[0]))
        reduced_data = model.fit_transform(np.asarray(df))

        metrics = visualization.desc_vis(algo, model, reduced_data, df)

    elif algo == 'Autoencoder':
        from keras.models import Model
        from keras.layers import Input, Dense, LeakyReLU

        input = Input(shape=df.shape[-1])
        enc = Dense(64)(input)
        enc = LeakyReLU()(enc)
        enc = Dense(32)(enc)
        enc = LeakyReLU()(enc)
        latent_space = Dense(int(hyper[0]), activation="tanh")(enc)
        dec = Dense(32)(latent_space)
        dec = LeakyReLU()(dec)
        dec = Dense(64)(dec)
        dec = LeakyReLU()(dec)
        dec = Dense(units=df.shape[-1], activation="sigmoid")(dec)

        autoencoder = Model(input, dec)
        autoencoder.compile(optimizer = "adam", metrics = ["mse"], loss = "mse")

        autoencoder.fit(df, df, epochs=50, batch_size=32, validation_split=0.25)
        model = Model(input, latent_space) #encoder
        reduced_data = model.predict(df)
        metrics = visualization.desc_vis(algo, model, reduced_data, df)

    return metrics

####################################

def train_clust_model(algo, hyper, data):
    df = data

    if algo == 'K-means Clustering':
        from sklearn.cluster import KMeans
        model = KMeans(n_clusters=int(hyper[0]), n_init='auto')
        labels = model.fit_predict(np.asarray(df))
        metrics = visualization.clust_vis(algo, model, labels, df)
    
    pickle.dump(model, open('./data/model.pickle', 'wb'))
    return metrics
