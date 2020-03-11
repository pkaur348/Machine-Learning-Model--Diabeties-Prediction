# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

df = pd.read_csv("diabetes.csv")


print(df.head())
print(df.columns)
print(df.info())
print(df.describe())
sb.pairplot(df)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import sklearn.cluster as cluster
from sklearn.externals import joblib

X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values

kmeans = cluster.KMeans(n_clusters=2, init="k-means++", max_iter=300, n_init=10, random_state=0)
yMeans = kmeans.fit_predict(X)

plt.scatter(X[yMeans==0, 0], X[yMeans==0, 1], color="r")
plt.scatter(X[yMeans==1, 0], X[yMeans==1, 1], color="g")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='y')
plt.show()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

classifier = DecisionTreeClassifier()

classifier.fit(X_train, Y_train)

predict = classifier.predict(X_test)

score   = cross_val_score(classifier, X, Y)
print(np.mean(score)*100)

mt      = confusion_matrix(Y_test, predict)
print(mt)

joblib.dump(classifier, 'predict.result')