# JLL, 2022.5.23 - 24
# (OP082) jinn@Liu:~/openpilot/selfdrive/controls/tests$ python test_clustering_tutor.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, cophenet, dendrogram, fcluster
from scipy.cluster import _hierarchy
from fastcluster import linkage_vector

  #===== from Jörn's Blog: SciPy Hierarchical Clustering and Dendrogram Tutorial
np.set_printoptions(precision=2, suppress=True)  # suppress scientific float notation
  # generate two clusters: a with 3 points, b with 2:
np.random.seed(4711)  # for repeatability of this tutorial
a = np.random.multivariate_normal([10, 0], [[2, 1], [1, 3]], size=[3,])
b = np.random.multivariate_normal([0, 10], [[2, 1], [1, 3]], size=[2,])
print("#--- b =", b)

X = np.concatenate((a, b),)
print("#--- X =", X)  # X.shape = (5, 2) = (m, n); 5 samples with 2 dimensions

Y = pdist(X)  # Y=dist(X[i], X[j]): condensed distance matrix [ij] with i < j < m (upper m x m matrix)
              # C^m_2 = m!/(2!*(m-2)!) = 5*4/2 = 10; stored in entry m*i + j - ((i+2)*(i+1))//2 = 10
print("#--- Y =", Y)

Z = linkage(X, 'centroid')  # Z.shape = (4=m-1, 4) linkage matrix; Perform hierarchical/agglomerative clustering.
print("#--- Z =", Z)  # See scipy.cluster.hierarchy.linkage; m-1 iterations; What is the formula of centroid?
  # A cluster with an index less than m corresponds to one of the m original observations.

CCorDst, CDstMtx = cophenet(Z, Y)  # CDstMtx.shape = (10,); wiki: Cophenetic correlation
print("#--- CCorDst =", CCorDst)
print("#--- CDstMtx =", CDstMtx)

  #===== from /openpilot/selfdrive/controls/tests/test_clustering.py
def OPfcluster(Z, t, criterion='inconsistent', depth=2, R=None, monocrit=None):
    # supersimplified function to get fast clustering. Got it from scipy
  Z = np.asarray(Z, order='c')
  n = Z.shape[0] + 1
  T = np.zeros((n,), dtype='i')
  _hierarchy.cluster_dist(Z, T, float(t), int(n))
  return T

OPcluster_idxs = OPfcluster(Z, 2.5, criterion='distance')  # t=2.5 threshold
  #OPcluster_idxs = OPfcluster(X, 2.5, criterion='distance')  # Error
SPcluster_idxs = fcluster(Z, 5, criterion='distance')
print("#--- OPcluster_idxs, SPcluster_idxs =", OPcluster_idxs, SPcluster_idxs)
print("#--- No. of OP clusters =", len(np.unique(OPcluster_idxs)) )

old_link = linkage_vector(X, method='centroid')
old_cluster_idxs = OPfcluster(old_link, 2.5, criterion='distance')
print("#--- old_link =", old_link)
print("#--- old_cluster_idxs =", old_cluster_idxs)

plt.clf()
plt.subplot(121)
plt.title('Sample Data')
plt.scatter(X[:,0], X[:,1])

plt.subplot(122)
  #plt.figure(figsize=(25, 10))  # plot full dendrogram
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    #leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=12.,  # font size for the x axis labels
)
plt.show()
  # Done. Do /openpilot/selfdrive/controls/tests/test_clustering.py
