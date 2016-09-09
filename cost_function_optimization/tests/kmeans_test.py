from sklearn.datasets import *
import numpy as np
import matplotlib.pyplot as plt
from cost_function_optimization import kmeans_clustering
from sequential import BSAS
from validity_scripts import internal_criteria, external_criteria, relative_criteria
from scipy.stats import norm
from tqdm import tqdm
from sys import maxsize as max_integer
from utility.plotting_functions import *

import unittest

plt.style.use('ggplot')

euclidean_distance = lambda data, point: np.sqrt(np.sum(np.power(data - point, 2), axis = 1).reshape((len(data), 1)))

class Test(unittest.TestCase):

    #@unittest.skip("no")
    def testBlobs(self):
        no_of_clusters = 8
        
        # Create the dataset
        X, y = make_blobs(n_samples = 1000, centers= no_of_clusters, n_features=2,random_state=11)
        
        # Run the clustering algorithm but first run a sequential algorithm to obtain initial centroids
        clustered_data, centroids, total_clusters = BSAS.basic_sequential_scheme(X)
        X, centroids, centroids_history = kmeans_clustering.kmeans(X, no_of_clusters, centroids_initial = centroids)

        # Plotting
        plot_data(X, centroids, no_of_clusters, centroids_history)
        
        # Examine Cluster Validity with statistical tests
        initial_gamma, list_of_gammas, result = internal_criteria.internal_validity(X, no_of_clusters, kmeans_clustering.kmeans)
        initial_indices, list_of_indices, result_list = external_criteria.external_validity(X, no_of_clusters, y, kmeans_clustering.kmeans)
        
        # Histogram of gammas from internal criteria 
        hist_internal_criteria(initial_gamma, list_of_gammas, result)
        hist_external_criteria(initial_indices, list_of_indices, result_list)
        
        plt.show()
    
    @unittest.skip("no")
    def testCircles(self):
        no_of_clusters = 2
        
        # Create the dataset
        X, y = make_circles(n_samples=300, shuffle = True, noise = 0.05, factor = 0.5, random_state = 10)
        
        # Run the clustering algorithm
        X, centroids, centroids_history = kmeans_clustering.kmeans(X, no_of_clusters)
        
        # Plotting
        plot_data(X, centroids, no_of_clusters, centroids_history)
        
        # Examine Cluster Validity with statistical tests
        initial_gamma, list_of_gammas, result = internal_criteria.internal_validity(X, no_of_clusters, kmeans_clustering.kmeans)
        initial_indices, list_of_indices, result_list = external_criteria.external_validity(X, no_of_clusters, y, kmeans_clustering.kmeans)
        
        # Histogram of gammas from internal and external criteria 
        hist_internal_criteria(initial_gamma, list_of_gammas, result)
        hist_external_criteria(initial_indices, list_of_indices, result_list)
        
        plt.show()
        
    @unittest.skip("no")
    def testMoons(self):
        no_of_clusters = 2
        
        # Create the dataset
        X, y = make_moons(n_samples=300, shuffle = True, noise = 0.1, random_state = 10)
        
        # Run the clustering algorithm
        X, centroids, centroids_history = kmeans_clustering.kmeans(X, no_of_clusters)
        
        # Plotting
        plot_data(X, centroids, no_of_clusters, centroids_history)
        
        # Examine Cluster Validity with statistical tests
        initial_gamma, list_of_gammas, result = internal_criteria.internal_validity(X, no_of_clusters, kmeans_clustering.kmeans)
        initial_indices, list_of_indices, result_list = external_criteria.external_validity(X, no_of_clusters, y, kmeans_clustering.kmeans)
        
        # Histogram of gammas from internal and external criteria 
        hist_internal_criteria(initial_gamma, list_of_gammas, result)
        hist_external_criteria(initial_indices, list_of_indices, result_list)
        
        plt.show()
    
    
    ################################################## Relative Criteria Clustering #########################
    
    @unittest.skip('no')
    def testRelativeBlobs(self):
        no_of_clusters= 4
        
        # Create the dataset
        X, y = make_blobs(n_samples=300, centers= no_of_clusters, n_features=2,random_state=103)
        
        # Successive executions of the clustering algorithm
        no_of_clusters_list, DI, DB, SI, GI = relative_criteria.relative_validity_hard(X, no_of_clusters)
        
        # Plot the indices
        plot_relative_criteria_hard(no_of_clusters_list, DI, DB, SI, GI)
        plt.show()       
    
    
                
                
                
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()