# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm, datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.DataFrame(iris.target)


# so let's make feature engineering to determining the most important

# iris_features = iris.data
# iris_type = iris.target

# lets visualize features to find most important

def plotSVC(title, svc, features):
    x_min, x_max = features[:, 0].min(), features[:, 0].max()
    y_min, y_max = features[:, 1].min() - 1, features[:, 1].max() + 1
    h = (x_max / x_min) / 100
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    plt.subplot(1, 1, 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap="magma_r", alpha=0.8)

    # Plot also the training points
    plt.scatter(features[:, 0], features[:, 1], c=y, cmap='magma_r')
    plt.xlabel(title + ' length')
    plt.ylabel(title + ' width')
    plt.xlim(xx.min(), xx.max())
    # plt.ylim(yy.min(), yy.max())
    # plt.xticks(())
    # plt.yticks(())
    plt.title(title)

    plt.show()


# iris_data_sepal = iris.data[:, :2]
iris_data_sepal = X[['sepal length (cm)', 'sepal width (cm)']]
svc_sepal = svm.SVC().fit(iris_data_sepal.values, y.values.ravel())
plotSVC("Sepal", svc=svc_sepal, features=iris_data_sepal.values)
sepal_score = cross_val_score(svc_sepal, iris_data_sepal, y.values.ravel(), cv=4)
print(str(sepal_score.mean()) + " - sepal")

# sepal w/l is quite good feature to define iris type (score 0.8067)

# iris_data_petal = iris.data[:, 2:4]
iris_data_petal = X[['petal length (cm)', 'petal width (cm)']]
svc_petal = svm.SVC().fit(iris_data_petal.values, y.values.ravel())
plotSVC("Petal", svc=svc_petal, features=iris_data_petal.values)
petal_score = cross_val_score(svc_petal, iris_data_petal, y.values.ravel(), cv=4)
print(str(petal_score.mean()) + " - petal")

# petal is even better then sepal (score: 0.9665)


baseline = LogisticRegression(multi_class="multinomial", max_iter=1000)
scores = cross_val_score(baseline, X, y.values.ravel(), cv=4)
print(str(scores.mean()) + " - Baseline")

# Baseline model we need to have a point of reference. I use the LogisticRegression algoritm
# I use the LogisticRegression: score aprox 0.97332
# baseline uses all 4 features to predict


kernels = ['linear', 'rbf', 'poly', 'sigmoid']
for kernel in kernels:
    svc = svm.SVC(kernel=kernel).fit(X, y.values.ravel())
    # print(cross_val_score(svc, iris_features, iris_type, cv=4).mean())
    # plotSVC('kernel=' + str(kernel), svc=svc)

# As we can see the most useless kernel is sigmoid
# In this case linear kernel have best performance (almost 98 percents)


gammas = [0.1, 0.3, 0.5, 0.7, 1, 10, 100]
for gamma in gammas:
    svc = svm.SVC(kernel='rbf', gamma=gamma).fit(X, y.values.ravel())
    # print(cross_val_score(svc, iris_features, iris_type, cv=4).mean())
# plotSVC('gamma=' + str(gamma), svc=svc)

# increasing gamma leads to overfit, most successful gamma for rbf is 0.5
# 0.1 for poly. Linear doesn't use gamma, sigmoid isn't effective

cs = [0.1, 0.3, 0.5, 0.7, 1, 1.3, 1.5]
for c in cs:
    svc = svm.SVC(kernel='rbf', C=c).fit(X, y.values.ravel())
# print(cross_val_score(svc, X, y.values.ravel(), cv=4).mean())
# plotSVC('c=' + str(c), svc=svc)

# most effective C-param (penalty) is 1 and 1.5 for rbf, 1.3 for linear
# and 0.5 for poly


# lets use most effective hyperparams combination
rbf = svm.SVC(kernel='rbf', gamma=0.5, C=1).fit(X, y.values.ravel())
poly = svm.SVC(kernel='poly', gamma=0.1, C=0.5).fit(X, y.values.ravel())
scores_rbf = cross_val_score(rbf, X, y.values.ravel(), cv=4)
scores_poly = cross_val_score(poly, X, y.values.ravel(), cv=4)
print(str(scores_rbf.mean()) + " " + str(scores_poly.mean()))
# plotSVC('rbf ', svc=rbf)
# plotSVC('poly ', svc=poly)
# hyperparams combining did not give a noticeable increase in accuracy

# let's add new features to increase score
X['sepal sum'] = iris_data_sepal.apply(np.sum, axis=1)
X['petal sum'] = iris_data_petal.apply(np.sum, axis=1)

iris_data = X[['sepal sum', 'petal sum']]
svc_iris = svm.SVC(kernel='poly', gamma=0.1, C=0.5).fit(iris_data.values, y.values.ravel())
plotSVC("Petal/Sepal", svc=svc_iris, features=iris_data.values)
petal_score = cross_val_score(svc_iris, iris_data, y.values.ravel(), cv=4)
print(str(petal_score.mean()) + " - petal to sepal")
# new features provide us the best score (> 0.98)

# optional Посчитать AUC для этой задачи. Почему это работает? А визуализировать получится?
# Используйте GridSearch и кросс-валидацию для подбора гипер-параметров
