from sklearn.datasets import *
import numpy as np
import matplotlib.pyplot as plt
from sequential import BSAS
from validity_scripts import internal_criteria, external_criteria, relative_criteria
from scipy.stats import norm
from tqdm import tqdm
from sys import maxsize as max_integer
from utility.plotting_functions import *

import unittest


class Test(unittest.TestCase):


    #@unittest.skip("no")
    def testBlobs(self):
        no_of_clusters = 6
        
        # Create the dataset
        X, y = make_blobs(n_samples = 1000, centers= no_of_clusters, n_features=2,random_state=None)
        
        # Run the clustering algorithm
        X, centroids, no_of_clusters = BSAS.basic_sequential_scheme(X)

        # Plotting
        plot_data(X, centroids, no_of_clusters)
        
        # Examine Cluster Validity with statistical tests
        #initial_gamma, list_of_gammas, result = internal_criteria.internal_validity(X, no_of_clusters, kmeans_clustering.kmeans)
        #initial_indices, list_of_indices, result_list = external_criteria.external_validity(X, no_of_clusters, y, kmeans_clustering.kmeans)
        
        # Histogram of gammas from internal criteria 
        #hist_internal_criteria(initial_gamma, list_of_gammas, result)
        #hist_external_criteria(initial_indices, list_of_indices, result_list)
        
        plt.show()


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()