import numpy as np
from itertools import permutations
from sklearn import datasets
from matplotlib import pyplot


def centroids_random_select(X, n_clusters):
    return X[np.random.choice(X.shape[0], size=n_clusters, replace=False), :]


def centroids_smart_select(X, n_clusters):
  """
  Input:
  X: numpy array, dimension (n, d) - dataset
  n_clusters: int - so cum (clusters)
  Output:
  centroids: numpy array, dimension (n_cluster, d)
            - cac centroid duoc chon de bat dau thuat toan
  """
  centroids = np.empty((n_clusters, X.shape[1]))
  centroids[0] = X[np.random.choice(X.shape[0])]
  for j in range(1, n_clusters):
    prob = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
      _, prob[i] = find_nearest_centroid(X[i], centroids[:j])
    prob /= prob.sum()
    centroids[j] = X[np.random.choice(X.shape[0], p=prob)]
  return centroids
  


def find_nearest_centroid(x, centroids):
  """
  Input:
  x: numpy array, dimension (d, ) - diem data
  centroids: numpy array, dimension (n_clusters, d) - cac centroid hien tai
  Output:
  label_x: int - so thu tu centroid gan x nhat (trong khoang 0...n_clusters)
  distance_x: float64 - khoang cach tu x den centroid gan x nhat
  """
  D = np.sum((x - centroids)**2, axis = 1)
  return np.argmin(D), np.amin(D)


def find_new_centroid(X, labels, j):
  """
  Input:
  X: numpy.ndarray, dimension (n, d) - dataset
  labels: numpy.ndarray, dimension (n, ) - array chua cluster hien tai cua tung diem data
          (Voi diem data X[i], labels[i] = so thu tu cua cluster dang chua X[i])
  j: int (trong khoang 0...n_clusters) - so thu tu cua cluster dang can update centroid
  Output:
  new_centroid_j: numpy.ndarray, dimension (d, ) - centroid moi cua cluster thu j
  """
  return np.mean(X[labels == j], axis=0)  # Day la output sai, hay sua lai cho dung


def my_kmeans(X, n_clusters, max_iter=100, smart=False):
    # Khoi tao cac centroid ban dau bang cach chon ngau nhien
    centroids = (
        centroids_smart_select(X, n_clusters) if smart
        else centroids_random_select(X, n_clusters)
    )
    labels = - np.ones(X.shape[0])  # khoi tao bang vector [-1, -1, ..., -1]
    for iteration in range(max_iter):
        # voi moi diem data X[i], tim centroid gan nhat cho X[i]
        for i in range(X.shape[0]):
            labels[i], _ = find_nearest_centroid(X[i], centroids)
        # voi moi cluster j, cap nhat lai centroid cho cluster j
        new_centroids = np.empty((n_clusters, X.shape[1]))
        for j in range(n_clusters):
            new_centroids[j] = find_new_centroid(X, labels, j)
        # neu cac centroid da cap nhat van bang cac centroid cu thi dung lai
        if np.sum((centroids - new_centroids) ** 2) < 1e-200:
            break
        else:
            centroids = new_centroids
    return centroids, labels
