import os 
os.environ['OMP_NUM_THREADS']="1"
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn import cluster
from sklearn import metrics
from sklearn import tree
from sklearn import linear_model
from sklearn import neighbors
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split as tts
import warnings
warnings.simplefilter("ignore", category=ConvergenceWarning)
print('準確率')
wine = load_wine()
X = pd.DataFrame(wine.data, columns = wine.feature_names)
target = pd.DataFrame(wine.target, columns = ["target"])
y = target["target"]

kmeans = cluster.KMeans(n_clusters= 3, random_state=10, n_init='auto')
kmeans.fit(X)
kmeans_pred_y=np.choose(kmeans.labels_,[1,0,2])
#---------------------------------------------

dtree = tree.DecisionTreeClassifier(max_depth = 3)
dtree.fit(X, y)

#---------------------------------------------
logistic = linear_model.LogisticRegression()
logistic.fit(X, y)

#---------------------------------------------

lm = linear_model.LinearRegression()
lm.fit(X, y)

#---------------------------------------------

XTrain, XTest, yTrain, yTest = tts(X, y, test_size = 0.33, random_state = 10)

krate = {}
for k in range(1, 55):
    knn = neighbors.KNeighborsClassifier(n_neighbors= k)
    knn.fit(XTrain, yTrain)
    krate[k] = knn.score(XTest, yTest)
    
for key, value in krate.items():
    if value == max(list(krate.values())):
        print('knn: ', round(value*100, 2),'%')
        break

    
treerate = {}
for t in range(1, 120):
    dtree = tree.DecisionTreeClassifier(max_depth= k)
    dtree.fit(XTrain, yTrain)
    treerate[t] = dtree.score(XTest, yTest)
    
for key, value in treerate.items():
    if value == max(list(treerate.values())):
        print('tree: ', round(value*100, 2),'%')
        break

print('Kmean: ', round(metrics.accuracy_score(y,kmeans_pred_y)*100,2),'%')
print('logistic: ', round(logistic.score(X, y)*100,2),'%')
print('linear: ', round(lm.score(X, y)*100,2),'%')
print('knn: ', round(knn.score(X, y)*100,2),'%')