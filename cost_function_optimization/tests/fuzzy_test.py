from sklearn.datasets import *
import numpy as np
from cost_function_optimization import fuzzy_clustering
from validity_scripts import internal_criteria, external_criteria, relative_criteria
from scipy.stats import norm
from tqdm import tqdm
from sys import maxsize as max_integer
from utility.plotting_functions import *
import matplotlib.pyplot as plt

import unittest

plt.style.use('ggplot')

euclidean_distance = lambda data, point: np.sqrt(np.sum(np.power(data - point, 2), axis = 1).reshape((len(data), 1)))

class Test(unittest.TestCase):

    #@unittest.skip("no")
    def testBlobs(self):
        no_of_clusters = 6
        
        # Create the dataset
        X, y = make_blobs(n_samples = 50000, centers= no_of_clusters, n_features=2,random_state=11)
        
        # Run the clustering algorithm
        X, centroids, ita, centroids_history, partition_matrix = fuzzy_clustering.fuzzy(X, no_of_clusters)
        
        # Plotting
        plot_data(X, centroids, no_of_clusters, centroids_history)
        
        # Examine Cluster Validity with statistical tests
        #initial_gamma, list_of_gammas, result = internal_criteria.internal_validity(X, no_of_clusters, fuzzy_clustering.fuzzy)
        #initial_indices, list_of_indices, result_list = external_criteria.external_validity(X, no_of_clusters, y, fuzzy_clustering.fuzzy)
        
        # Histogram of gammas from internal criteria 
        #hist_internal_criteria(initial_gamma, list_of_gammas, result)
        #hist_external_criteria(initial_indices, list_of_indices, result_list)
        
        plt.show()
    
    @unittest.skip("no")
    def testCircles(self):
        no_of_clusters = 2
        
        # Create the dataset
        X, y = make_circles(n_samples=300, shuffle = True, noise = 0.05, factor = 0.5, random_state = 10)
        
        # Run the clustering Algorithm
        X, centroids, ita, centroids_history, partition_matrix = fuzzy_clustering.fuzzy(X, 2)
        
        # Plotting
        plot_data(X, centroids, no_of_clusters, centroids_history)
        
        # Examine Cluster Validity with statistical tests
        initial_gamma, list_of_gammas, result = internal_criteria.internal_validity(X, no_of_clusters , fuzzy_clustering.fuzzy)
        initial_indices, list_of_indices, result_list = external_criteria.external_validity(X, no_of_clusters, y, fuzzy_clustering.fuzzy)
        
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
        X, centroids, ita, centroids_history, partition_matrix = fuzzy_clustering.fuzzy(X, no_of_clusters)
        
        # Plotting
        plot_data(X, centroids, no_of_clusters, centroids_history)
        
        # Examine Cluster Validity with statistical tests
        initial_gamma, list_of_gammas, result = internal_criteria.internal_validity(X, no_of_clusters, fuzzy_clustering.fuzzy)
        initial_indices, list_of_indices, result_list = external_criteria.external_validity(X, no_of_clusters, y, fuzzy_clustering.fuzzy)
        
        # Histogram of gammas from internal and external criteria 
        hist_internal_criteria(initial_gamma, list_of_gammas, result)
        hist_external_criteria(initial_indices, list_of_indices, result_list)
        
        plt.show()
    
    
    ######################### Relative Criteria Clustering #########################
    
    @unittest.skip('no')
    def testRelativeBlobs(self):
        no_of_clusters= 5
        
        # Create the dataset
        X, y = make_blobs(n_samples=100, centers= no_of_clusters, n_features=2,random_state=20)
        
        # Successive executions of the clustering algorithm
        no_of_clusters_list, values_of_q, PC, PE, XB, FS = relative_criteria.relative_validity_fuzzy(X, no_of_clusters)
        
        # Plot the indices
        plot_relative_criteria_fuzzy(no_of_clusters_list, values_of_q, PC, PE, XB, FS)
        plt.show()        
    
    
    @unittest.skip('no')
    def testRelativeCircles(self):
        no_of_clusters= 2
        
        # Create the dataset
        X, y = make_circles(n_samples=100, shuffle = True, noise = 0.05, factor = 0.5, random_state = 10)
        
        # Successive executions of the clustering algorithm
        no_of_clusters_list, values_of_q, PC, PE, XB, FS = relative_criteria.relative_validity_fuzzy(X, no_of_clusters)
        
        # Plot the indices
        plot_relative_criteria_fuzzy(no_of_clusters_list, values_of_q, PC, PE, XB, FS)
        plt.show()      
    
    
    @unittest.skip('no')
    def testRelativeMoons(self):
        no_of_clusters= 2
        
        # Create the dataset
        X, y = make_moons(n_samples=100, shuffle = True, noise = 0.1, random_state = 10)
        
        # Successive executions of the clustering algorithm
        no_of_clusters_list, values_of_q, PC, PE, XB, FS = relative_criteria.relative_validity_fuzzy(X, no_of_clusters)
        
        # Plot the indices
        plot_relative_criteria_fuzzy(no_of_clusters_list, values_of_q, PC, PE, XB, FS)
        plt.show()      
    
    
    
      

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()