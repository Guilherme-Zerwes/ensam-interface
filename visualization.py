import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error,confusion_matrix, silhouette_score, davies_bouldin_score

plt.rcParams['figure.figsize'] = (6,3)
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['axes.grid'] = True
plt.rcParams['font.family'] = 'sans-serif'

def reg_vis(algorithm, model, X, Y, df, **kwargs):
    
    # x_train = X[0]
    x_test = X

    # y_train = Y[0]
    y_test = Y

    y_pred = model.predict(x_test)

    res = y_test - y_pred

    mae = f'Mean absolute error: {mean_absolute_error(y_test, y_pred) :.4f}'
    mse = f'Mean squared error: {mean_squared_error(y_test, y_pred) :.4f}'
    rmse = f'Root mean squared error: {np.sqrt(mean_squared_error(y_test, y_pred)) :.4f}'
    r2 = f'R² score: {model.score(x_test, y_test) :.4f}'
    
    #residue scatter plot
    # res_scatter_fig, res_scatter_ax = plt.subplots()
    # res_scatter_ax.scatter(y_pred, res)
    # res_scatter_ax.axhline(y=np.mean(res), color='r', ls='--', linewidth=2)

    #histogram residue
    res_hist_fig, res_hist_ax = plt.subplots()
    res_hist_ax.set_title('Residals Histogram')
    res_hist_ax.set_xlabel('Residual')
    res_hist_ax.set_ylabel('Relative Frequency')
    res_hist_fig.tight_layout()
    sns.histplot(data=res, kde=True)

    #predicted x real
    compare_fig, compare_ax = plt.subplots()
    compare_ax.set_title('Predicted x Actual')
    compare_ax.set_xlabel('Real Y values')
    compare_ax.set_ylabel('Predicted Y values')
    compare_fig.tight_layout()
    sns.scatterplot(x=y_test, y=y_pred)

    metrics = [mae, mse, rmse, r2, res_hist_fig, compare_fig]
    if algorithm == 'Linear Regression':
        #coefficiets plot
        coef_fig, coef_ax = plt.subplots()
        coef_ax.set_title('Coefficients Plot')
        coef_ax.set_xlabel('Coefficient')
        coef_ax.set_ylabel('Values')
        coef_fig.tight_layout()
        sns.barplot(x=[i for i in range(len(list(df[0].keys())))], y=model.coef_)

        metrics.append(coef_fig)

    elif algorithm == 'Random Forests Regressor':
        from sklearn import tree
        #Random forrest single tree plot
        fn=df[0].keys()
        cn=df[1].keys()
        fig_tree, axes_tree = plt.subplots(dpi=200)
        axes_tree.set_title('Decision Tree')
        tree.plot_tree(model.estimators_[0], feature_names = fn, class_names=cn, filled = True)
        metrics.append(fig_tree)

        #Feature importance
        importances = model.feature_importances_
        std_importance = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
        relevants = importances > 0.05
        forest_importances = pd.Series(importances[relevants], index=df[0].keys()[relevants])

        fig_importance, ax_importance = plt.subplots()
        forest_importances.plot.bar(yerr=std_importance[relevants], ax=ax_importance)
        ax_importance.set_title("Feature importances")
        ax_importance.set_ylabel("Mean decrease in impurity")
        ax_importance.tick_params(axis='both', which='major', labelsize=3)
        fig_importance.tight_layout()

        metrics.append(fig_importance)

    elif algorithm == 'Support Vector Machines Regression':
        fig_support, ax_support = plt.subplots()
        ax_support.scatter(x_test[:,0], x_test[:,1])
        ax_support.scatter(model.support_vectors_[:,0], model.support_vectors_[:,1], color='r')
        ax_support.set_title('Support Vectors')
        fig_support.tight_layout()
        
        metrics.append(fig_support)


    return metrics

##################################################

def class_vis(algorithm, model, X, Y, df, **kwargs):
    
    # x_train = X[0]
    x_test = X

    # y_train = Y[0]
    y_test = Y

    y_pred = model.predict(x_test)

    #Numeric metrics
    acc = f'Accuracy: {accuracy_score(y_test, y_pred) :.4f}'

    #Confusion matrix
    conf_fig, conf_ax = plt.subplots()
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    sns.heatmap(data=cm, fmt='g')
    conf_ax.set_title('Confusion Matrix')
    conf_ax.set_ylabel('True Labels')
    conf_ax.set_xlabel('Predicted Labels')

    metrics = [acc, conf_fig]

    if algorithm == 'Decision Tree Classifier':
        from sklearn import tree

        #Random forrest single tree plot
        fn = df[0].keys()

        fig_tree, axes_tree = plt.subplots(dpi=300)
        axes_tree.set_title('Decision Tree')
        tree.plot_tree(model, feature_names=fn, class_names=True, filled = True)
        metrics.append(fig_tree)

        #Feature importance
        importances = model.feature_importances_
        std_importance = np.std(importances)
        relevants = importances > 0.05
        forest_importances = pd.Series(importances[relevants], index=df[0].keys()[relevants])

        fig_importance, ax_importance = plt.subplots()
        forest_importances.plot.bar(yerr=std_importance, ax=ax_importance)
        ax_importance.set_title("Feature importances")
        ax_importance.set_ylabel("Mean decrease in impurity")
        ax_importance.tick_params(axis='both', which='major', labelsize=3)
        fig_importance.tight_layout()

        metrics.append(fig_importance)

    elif algorithm == 'Random Forests Classifier':
        from sklearn import tree

        #Random forrest single tree plot
        fn = df[0].keys()
        
        fig_tree, axes_tree = plt.subplots(dpi=300)
        axes_tree.set_title('Decision Tree')
        tree.plot_tree(model.estimators_[0], feature_names=fn, class_names=True, filled = True)
        metrics.append(fig_tree)

        #Feature importance
        importances = model.feature_importances_
        std_importance = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
        relevants = importances > 0.05
        forest_importances = pd.Series(importances[relevants], index=df[0].keys()[relevants])
        
        fig_importance, ax_importance = plt.subplots()
        forest_importances.plot.bar(yerr=std_importance[relevants], ax=ax_importance)
        ax_importance.set_title("Feature importances")
        ax_importance.set_ylabel("Mean decrease in impurity")
        ax_importance.tick_params(axis='both', which='major', labelsize=3)
        fig_importance.tight_layout()

        metrics.append(fig_importance)

    elif algorithm == 'Support Vector Machine Classifier':
        fig_support, ax_support = plt.subplots()
        ax_support.scatter(x_test[:,0], x_test[:,1])
        ax_support.scatter(model.support_vectors_[:,0], model.support_vectors_[:,1], color='r')
        ax_support.set_title('Support Vectors')
        ax_support.set_xlabel('Feature 1')
        ax_support.set_ylabel('Feature 2')
        fig_support.tight_layout()
        
        metrics.append(fig_support)

    return metrics

##################################################

def desc_vis(algorithm, model, data, df, **kwargs):
    metrics = ['']
    if algorithm == 'Principal Component Analysis':
    
        n_components = data.shape[-1]

        #scree plot
        ind = np.arange(0, n_components)
        fig_scree, ax_scree = plt.subplots()
        sns.pointplot(x=ind, y=model.explained_variance_ratio_)
        ax_scree.set_title('Scree plot')
        ax_scree.set_xticks(ind)
        ax_scree.set_xticklabels(ind + 1)
        ax_scree.set_xlabel('Component Number')
        ax_scree.set_ylabel('Explained Variance')
        
        metrics.append(fig_scree)

        # Correlation Circle
        fig_cor, ax_cor = plt.subplots(figsize=(8,8))
        for i in range(0, model.components_.shape[1]):
            ax_cor.arrow(0,
                    0,  # Start the arrow at the origin
                    model.components_[0, i],  #0 for PC1
                    model.components_[1, i],  #1 for PC2
                    head_width=0.05,
                    head_length=0.1)

            plt.text(model.components_[0, i] + 0.05,
                    model.components_[1, i] + 0.05,
                    df.columns.values[i])


        an = np.linspace(0, 2 * np.pi, 100)
        plt.plot(np.cos(an), np.sin(an))  # Add a unit circle for scale
        plt.axis('equal')
        ax_cor.set_title('Variable factor map')
        metrics.append(fig_cor)


    elif algorithm == 'Kernel Principal Component Analysis':
        n_components = data.shape[-1]

        # Correlation Circle
        fig_cor, ax_cor = plt.subplots(figsize=(8,8))
        for i in range(0, model.eigenvectors_.shape[1]):
            ax_cor.arrow(0,
                    0,  # Start the arrow at the origin
                    model.eigenvectors_[0, i],  #0 for PC1
                    model.eigenvectors_[1, i],  #1 for PC2
                    head_width=0.05,
                    head_length=0.1)

            plt.text(model.eigenvectors_[0, i] + 0.05,
                    model.eigenvectors_[1, i] + 0.05,
                    df.columns.values[i])


        an = np.linspace(0, 2 * np.pi, 100)
        plt.plot(np.cos(an), np.sin(an))  # Add a unit circle for scale
        plt.axis('equal')
        ax_cor.set_title('Variable factor map')
        metrics.append(fig_cor)

    elif algorithm == 'Autoencoder':
        if data.shape[-1] == 2:
            fig_red, ax_red = plt.subplots()
            sns.scatterplot(x=data[:,0], y=data[:,1])
            metrics.append(fig_red)
    return metrics

##################################################

def clust_vis(algorithm, model, label, df, **kwargs):
    metrics = []
    if algorithm == 'K-means Clustering':
        silhoette = silhouette_score(X=np.asarray(df), labels=label)
        davies = davies_bouldin_score(X=np.asarray(df), labels=label)
        
        metrics.append(f'Silhouette score: {silhoette :.4f}')
        metrics.append(f'Davies Bouldin score: {davies :.4f}')
    return metrics