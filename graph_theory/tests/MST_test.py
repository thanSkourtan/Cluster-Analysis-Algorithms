from graph_theory import MST
from sklearn.datasets import *
import numpy as np
import matplotlib.pyplot as plt
from validity_scripts import internal_criteria, external_criteria, relative_criteria
from scipy.stats import norm
from tqdm import tqdm
from sys import maxsize as max_integer
from utility.plotting_functions import *

import unittest

plt.style.use('ggplot')

euclidean_distance = lambda data, point: np.sqrt(np.sum(np.power(data - point, 2), axis = 1).reshape((len(data), 1)))

class Test(unittest.TestCase):

    @unittest.skip("no")
    def testBlobs(self):
        no_of_clusters = 4
        
        # Create the dataset
        X, y = make_blobs(n_samples = 500, centers= no_of_clusters, n_features=2,random_state=121)
        
        # Run the clustering algorithm but first run a sequential algorithm to obtain initial centroids
        X, no_of_clusters = MST.minimum_spanning_tree(X, k = 3, f = 3.5)
        
        # Plotting
        plot_data(X, no_of_clusters)
        '''
        # Examine Cluster Validity with statistical tests
        initial_gamma, list_of_gammas, result = internal_criteria.internal_validity(X, no_of_clusters, MST.minimum_spanning_tree)
        initial_indices, list_of_indices, result_list = external_criteria.external_validity(X, no_of_clusters, y, MST.minimum_spanning_tree)
        
        # Histogram of gammas from internal criteria 
        hist_internal_criteria(initial_gamma, list_of_gammas, result)
        hist_external_criteria(initial_indices, list_of_indices, result_list)
        '''
        plt.show()
    
    #@unittest.skip("no")
    def testCircles(self):
        # Create the dataset
        X, y = make_circles(n_samples=500, shuffle = True, noise = 0.05, factor = 0.5, random_state = 118)
        
        # Run the clustering algorithm
        X, no_of_clusters = MST.minimum_spanning_tree(X, k = 7, f = 2.7)
        
        # Plotting
        plot_data(X, no_of_clusters)
        '''
        # Examine Cluster Validity with statistical tests
        initial_gamma, list_of_gammas, result = internal_criteria.internal_validity(X, no_of_clusters, MST.minimum_spanning_tree)
        initial_indices, list_of_indices, result_list = external_criteria.external_validity(X, no_of_clusters, y, MST.minimum_spanning_tree)
        
        # Histogram of gammas from internal and external criteria 
        hist_internal_criteria(initial_gamma, list_of_gammas, result)
        hist_external_criteria(initial_indices, list_of_indices, result_list)
        '''
        plt.show()
        
    @unittest.skip("no")
    def testMoons(self):
        # Create the dataset
        X, y = make_moons(n_samples=500, shuffle = True, noise = 0.07, random_state = 10)
        
        # Run the clustering algorithm
        X, no_of_clusters = MST.minimum_spanning_tree(X, k = 3, f = 2.7)
        
        # Plotting
        plot_data(X, no_of_clusters)
        
        # Examine Cluster Validity with statistical tests
        initial_gamma, list_of_gammas, result = internal_criteria.internal_validity(X, no_of_clusters, MST.minimum_spanning_tree)
        initial_indices, list_of_indices, result_list = external_criteria.external_validity(X, no_of_clusters, y, MST.minimum_spanning_tree)
        
        # Histogram of gammas from internal and external criteria 
        hist_internal_criteria(initial_gamma, list_of_gammas, result)
        hist_external_criteria(initial_indices, list_of_indices, result_list)
        
        plt.show()
    
    
    ################################################## Relative Criteria Clustering #########################
    
    @unittest.skip('no')
    def testRelativeBlobs(self):
        no_of_clusters= 4
        
        # Create the dataset
        X, y = make_blobs(n_samples=500, centers= no_of_clusters, n_features=2,random_state=121)
        
        # Successive executions of the clustering algorithm
        no_of_k_list, no_of_f_list, DI, SI= relative_criteria.relative_validity_hard_graph(X)

        # Plot the indices
        plot_relative_criteria_graph(no_of_k_list, no_of_f_list, DI, SI)
        plt.show()         
    
    @unittest.skip('no')
    def testRelativeCircles(self):

        # Create the dataset
        X, y = make_circles(n_samples=500, shuffle = True, noise = 0.05, factor = 0.5, random_state = 118)
        
        # Successive executions of the clustering algorithm
        no_of_k_list, no_of_f_list, DI, SI= relative_criteria.relative_validity_hard_graph(X)

        # Plot the indices
        plot_relative_criteria_graph(no_of_k_list, no_of_f_list, DI, SI)
        plt.show()     
    
    
    @unittest.skip('no')
    def testRelativeMoons(self):

        # Create the dataset
        X, y = make_moons(n_samples=500, shuffle = True, noise = 0.07, random_state = 118)
        
        # Successive executions of the clustering algorithm
        no_of_k_list, no_of_f_list, DI, SI= relative_criteria.relative_validity_hard_graph(X)

        # Plot the indices
        plot_relative_criteria_graph(no_of_k_list, no_of_f_list, DI, SI)
        plt.show()  
                
                
                
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()