from sklearn.datasets import *
import numpy as np
import matplotlib.pyplot as plt
from cost_function_optimization import fuzzy_clustering, possibilistic_clustering
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
        no_of_clusters = 3
        
        # Create the dataset
        X, y = make_blobs(n_samples = 5000, centers= no_of_clusters, n_features=2,random_state=None)
        
        # Run the clustering algorithm. First run fuzzy clustering to get ita and centroids
        X_, centroids, ita, centroids_history, partition_matrix = fuzzy_clustering.fuzzy(X, no_of_clusters)
        X, centroids, centroids_history, typicality_matrix = possibilistic_clustering.possibilistic(X, no_of_clusters, ita, centroids_initial = centroids)

        # Plotting
        plot_data(X, no_of_clusters, centroids, centroids_history)
        
        # Examine Cluster Validity with statistical tests
        initial_gamma, list_of_gammas, result = internal_criteria.internal_validity(X, no_of_clusters, possibilistic_clustering.possibilistic)
        initial_indices, list_of_indices, result_list = external_criteria.external_validity(X, no_of_clusters, y,  possibilistic_clustering.possibilistic)
        
        # Histogram of gammas from internal criteria 
        hist_internal_criteria(initial_gamma, list_of_gammas, result)
        hist_external_criteria(initial_indices, list_of_indices, result_list)
        
        plt.show()
        
    @unittest.skip("no")
    def testCircles(self):
        no_of_clusters = 2
        
        # Create the dataset
        X, y = make_circles(n_samples=300, shuffle = True, noise = 0.03, factor = 0.5, random_state = 10)
        
        # Run the clustering Algorithm
        X_, centroids, ita, centroids_history, partition_matrix = fuzzy_clustering.fuzzy(X, no_of_clusters)
        X, centroids, centroids_history, typicality_matrix = possibilistic_clustering.possibilistic(X, no_of_clusters, ita, centroids_initial = centroids)
        
        # Plotting
        plot_data(X, centroids, no_of_clusters, centroids_history)
        
        # Examine Cluster Validity with statistical tests
        initial_gamma, list_of_gammas, result = internal_criteria.internal_validity(X, no_of_clusters, possibilistic_clustering.possibilistic)
        initial_indices, list_of_indices, result_list = external_criteria.external_validity(X, no_of_clusters, y,  possibilistic_clustering.possibilistic)
        
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
        X_, centroids, ita, centroids_history, partition_matrix =fuzzy_clustering.fuzzy(X, no_of_clusters)
        X, centroids, centroids_history, typicality_matrix = possibilistic_clustering.possibilistic(X, no_of_clusters, ita, centroids_initial = centroids)
        
        # Plotting
        plot_data(X, centroids, no_of_clusters, centroids_history)
        
        # Examine Cluster Validity with statistical tests
        initial_gamma, list_of_gammas, result = internal_criteria.internal_validity(X, no_of_clusters, possibilistic_clustering.possibilistic)
        initial_indices, list_of_indices, result_list = external_criteria.external_validity(X, no_of_clusters, y,  possibilistic_clustering.possibilistic)
        
        # Histogram of gammas from internal and external criteria 
        hist_internal_criteria(initial_gamma, list_of_gammas, result)
        hist_external_criteria(initial_indices, list_of_indices, result_list)
        
        plt.show()
    
    
    ######################### Relative Criteria Clustering #########################
    
    # Although the code below works, we will not calculate relative indices for possibilistic 
    # clustering.
    # No relative indices for possibilistic
    '''
    @unittest.skip('no')
    def testRelativeBlobs(self):
        no_of_clusters= 5
        
        # Create the dataset
        X, y = make_blobs(n_samples=100, centers= no_of_clusters, n_features=2,random_state=20)
        
        # Successive executions of the clustering algorithm
        no_of_clusters_list, values_of_q, PC, PE, XB, FS = relative_criteria.relative_validity(X, no_of_clusters, possibilistic_clustering.possibilistic)
        
        # Plot the indices
        plot_relative_criteria(no_of_clusters_list, values_of_q, PC, PE, XB, FS)
        plt.show()        
    
    
    @unittest.skip('no')
    def testRelativeCircles(self):
        no_of_clusters= 2
        
        # Create the dataset
        X, y = make_circles(n_samples=100, shuffle = True, noise = 0.05, factor = 0.5, random_state = 10)
        
        # Successive executions of the clustering algorithm
        no_of_clusters_list, values_of_q, PC, PE, XB, FS = relative_criteria.relative_validity(X, no_of_clusters, possibilistic_clustering.possibilistic)
        
        # Plot the indices
        plot_relative_criteria(no_of_clusters_list, values_of_q, PC, PE, XB, FS)
        plt.show()      
    
    
    @unittest.skip('no')
    def testRelativeMoons(self):
        no_of_clusters= 2
        
        # Create the dataset
        X, y = make_moons(n_samples=100, shuffle = True, noise = 0.1, random_state = 10)
        
        # Successive executions of the clustering algorithm
        no_of_clusters_list, values_of_q, PC, PE, XB, FS = relative_criteria.relative_validity(X, no_of_clusters, possibilistic_clustering.possibilistic)
        
        # Plot the indices
        plot_relative_criteria(no_of_clusters_list, values_of_q, PC, PE, XB, FS)
        plt.show() 
        '''
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()