import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
from scipy.stats import mode

iris = load_iris()
X = iris.data
y_true = iris.target

def get_cluster_accuracy(y_true, y_pred):
    labels = np.zeros_like(y_pred)

    for i in range(3):
        mask = (y_pred == i)
        if np.sum(mask) == 0:
            continue
        labels[mask] = mode(y_true[mask])[0]

    return accuracy_score(y_true, labels)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X)
kmeans_acc = get_cluster_accuracy(y_true, kmeans_labels)


gmm = GaussianMixture(n_components=3, random_state=42)
gmm_labels = gmm.fit_predict(X)
gmm_acc = get_cluster_accuracy(y_true, gmm_labels)


print("K-Means Accuracy:", kmeans_acc * 100, "%")
print("EM (GMM) Accuracy:", gmm_acc * 100, "%")


if kmeans_acc > gmm_acc:
    print("K-Means performs better")
elif gmm_acc > kmeans_acc:
    print("EM (GMM) performs better")
else:
    print("Both perform equally")
