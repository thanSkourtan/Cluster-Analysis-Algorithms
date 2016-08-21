from sklearn.datasets import *
import numpy as np
import matplotlib.pyplot as plt
from cost_function_optimization import fuzzy_clustering


import unittest

plt.style.use('ggplot')

class Test(unittest.TestCase):

    @unittest.skip("no")
    def testBlobs(self):
        no_of_clusters= 5
        X, y = make_blobs(n_samples=1000, centers= no_of_clusters, n_features=2,random_state=10)
        X, centroids, ita, centroids_history = fuzzy_clustering.fuzzy(X, no_of_clusters)
        plot_data_util(X, centroids, centroids_history, no_of_clusters)
        
    @unittest.skip("no")
    def testCircles(self):
        no_of_clusters = 2
        X, y = make_circles(n_samples=1000, shuffle = True, noise = 0.05, factor = 0.5, random_state = 10)
        X, centroids, ita, centroids_history = fuzzy_clustering.fuzzy(X, 2)
        plot_data_util(X, centroids, centroids_history, no_of_clusters)
        
    #@unittest.skip("no")
    def testMoons(self):
        no_of_clusters = 2
        X, y = make_moons(n_samples=1000, shuffle = True, noise = 0.1, random_state = 10)
        X, centroids, ita, centroids_history = fuzzy_clustering.fuzzy(X, 2)
        X = plot_data_util(X, centroids, centroids_history, no_of_clusters)
        
    
    
def plot_data_util(X, centroids, centroids_history ,no_of_clusters):
    np.random.seed(seed = 13)
    clusters = np.unique(X[:, 2])
    
    for i, cluster in enumerate(clusters):
        plt.scatter(X[ X[:,2] == cluster, 0], X[ X[:, 2] == cluster, 1], c=np.random.rand(3,1), s = 30)
    
    # Plots the centroids history
    colors= ['k', 'b', 'g', 'y', 'm']
    for i in range(0, len(centroids_history),  no_of_clusters):
        for j in range(i, i + no_of_clusters):
            plt.plot(centroids_history[j, 0], centroids_history[j, 1], c = colors[j % no_of_clusters], marker = 'x', mew =  1, ms = 10)
    
    # Plots the centroids
    for i, c in enumerate(centroids):
            plt.plot(centroids[i, 0], centroids[i, 1], c = 'r', marker = 'x', mew=2, ms = 10)
    


    plt.show()


def evaluate():
    pass


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()