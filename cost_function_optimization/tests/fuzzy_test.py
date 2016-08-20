from sklearn.datasets import *
import numpy as np
import matplotlib.pyplot as plt
from cost_function_optimization import fuzzy_clustering


import unittest


class Test(unittest.TestCase):

    #@unittest.skip("no")
    def testBlobs(self):
        X, y = make_blobs(n_samples=1000, centers=3, n_features=2,random_state=10)
        X, centroids, ita = fuzzy_clustering.fuzzy(X, 3)
        plot_data_util(X, centroids)
        
    #@unittest.skip("no")
    def testCircles(self):
        X, y = make_circles(n_samples=1000, shuffle = True, noise = 0.05, factor = 0.5, random_state = 10)
        X, centroids, ita = fuzzy_clustering.fuzzy(X, 2)
        plot_data_util(X, centroids)
        
    #@unittest.skip("no")
    def testMoons(self):
        X, y = make_moons(n_samples=1000, shuffle = True, noise = 0.1, random_state = 10)
        X, centroids, ita = fuzzy_clustering.fuzzy(X, 2)
        X = plot_data_util(X, centroids)
        
    
    
def plot_data_util(X, centroids):
    np.random.seed(seed = 10)
    clusters = np.unique(X[:, 2])
    
    for i, cluster in enumerate(clusters):
        plt.scatter(X[ X[:,2] == cluster, 0], X[ X[:, 2] == cluster, 1], c=np.random.rand(3,1), s = 30)
    
    # Plots the centroids
    for i, c in enumerate(centroids):
            plt.plot(centroids[i, 0], centroids[i, 1], c = 'r', marker = 'x', mew=2, ms = 10)

    plt.show()


def evaluate():
    pass


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()