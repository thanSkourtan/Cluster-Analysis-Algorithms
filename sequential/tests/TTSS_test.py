from sklearn.datasets import *
import numpy as np
import matplotlib.pyplot as plt
from sequential import TTSS
from validity_scripts import internal_criteria, external_criteria, relative_criteria
from scipy.stats import norm
from tqdm import tqdm
from sys import maxsize as max_integer
from utility.plotting_functions import *

import unittest


class Test(unittest.TestCase):


    @unittest.skip("no")
    def testBlobs(self):
        no_of_clusters = 6
        
        # Create the dataset
        X, y = make_blobs(n_samples = 200, centers= no_of_clusters, n_features=2,random_state=49)
        
        # Run the clustering algorithm
        X, centroids, no_of_clusters = TTSS.two_threshold_sequential_scheme(X)

        # Plotting
        plot_data(X, centroids, no_of_clusters)
        
        # Examine Cluster Validity with statistical tests
        initial_gamma, list_of_gammas, result = internal_criteria.internal_validity(X, no_of_clusters, TTSS.two_threshold_sequential_scheme)
        initial_indices, list_of_indices, result_list = external_criteria.external_validity(X, no_of_clusters, y, TTSS.two_threshold_sequential_scheme)
        
        # Histogram of gammas from internal criteria 
        hist_internal_criteria(initial_gamma, list_of_gammas, result)
        hist_external_criteria(initial_indices, list_of_indices, result_list)
        
        plt.show()
        
    
    @unittest.skip("no")
    def testCircles(self):
        no_of_clusters = 2
        
        # Create the dataset
        X, y = make_circles(n_samples=300, shuffle = True, noise = 0.05, factor = 0.5, random_state = 10)
        
        # Run the clustering Algorithm
        X, centroids, no_of_clusters = TTSS.two_threshold_sequential_scheme(X)
        
        # Plotting
        plot_data(X, centroids, no_of_clusters)
        
        # Examine Cluster Validity with statistical tests
        initial_gamma, list_of_gammas, result = internal_criteria.internal_validity(X, no_of_clusters , TTSS.two_threshold_sequential_scheme)
        initial_indices, list_of_indices, result_list = external_criteria.external_validity(X, no_of_clusters, y, TTSS.two_threshold_sequential_scheme)
        
        # Histogram of gammas from internal and external criteria 
        hist_internal_criteria(initial_gamma, list_of_gammas, result)
        hist_external_criteria(initial_indices, list_of_indices, result_list)
        
        plt.show()
        
    #@unittest.skip("no")
    def testMoons(self):
        no_of_clusters = 2
        
        # Create the dataset
        X, y = make_moons(n_samples=300, shuffle = True, noise = 0.1, random_state = 10)
        
        # Run the clustering algorithm
        X, centroids, no_of_clusters = TTSS.two_threshold_sequential_scheme(X)
        
        # Plotting
        plot_data(X, centroids, no_of_clusters)
        
        # Examine Cluster Validity with statistical tests
        initial_gamma, list_of_gammas, result = internal_criteria.internal_validity(X, no_of_clusters, TTSS.two_threshold_sequential_scheme)
        initial_indices, list_of_indices, result_list = external_criteria.external_validity(X, no_of_clusters, y, TTSS.two_threshold_sequential_scheme)
        
        # Histogram of gammas from internal and external criteria 
        hist_internal_criteria(initial_gamma, list_of_gammas, result)
        hist_external_criteria(initial_indices, list_of_indices, result_list)
        
        plt.show()
    


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()