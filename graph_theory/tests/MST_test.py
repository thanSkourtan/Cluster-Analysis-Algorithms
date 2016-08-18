from sklearn.datasets import *
import numpy as np
import matplotlib.pyplot as plt
from graph_theory import MST

import unittest


class Test(unittest.TestCase):


    def setUp(self):
        pass


    def tearDown(self):
        pass


    def testBlobs(self):
        X, y = make_blobs(n_samples=100, centers=3, n_features=2,random_state=10)
        X = MST.minimum_spanning_tree(X)
        plot_data_util(X)
    
    def testCircles(self):
        X, y = make_circles(n_samples=100, shuffle = True, noise = 0.05, factor = 0.5)
        X = MST.minimum_spanning_tree(X)
        plot_data_util(X)
    
    
    def testMoons(self):
        X, y = make_moons(n_samples=100, shuffle = True, noise = 0.1)
        X = MST.minimum_spanning_tree(X)
        X = plot_data_util(X)
        
    
    
def plot_data_util(X):
    clusters = np.unique(X[:, 2])
    colors = ['r', 'b','g', 'm', 'y', 'c', 'k','w']
    for i, cluster in enumerate(clusters):
        plt.scatter(X[ X[:,2] == cluster, 0], X[ X[:, 2] == cluster, 1], c = colors[i % 8], s = 30)
    plt.show()









if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()