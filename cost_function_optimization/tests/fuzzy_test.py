from sklearn.datasets import *
import numpy as np
import matplotlib.pyplot as plt
from cost_function_optimization import fuzzy_clustering
from validity_scripts import internal_criteria, external_criteria
from scipy.stats import norm
from tqdm import tqdm
from sys import maxsize as max_integer

import unittest

plt.style.use('ggplot')

euclidean_distance = lambda data, point: np.sqrt(np.sum(np.power(data - point, 2), axis = 1).reshape((len(data), 1)))

class Test(unittest.TestCase):

    @unittest.skip("no")
    def testBlobs(self):
        no_of_clusters = 6
        
        # Create the dataset
        X, y = make_blobs(n_samples = 100, centers= no_of_clusters, n_features=2,random_state=11)
        
        # Run the clustering algorithm
        X, centroids, ita, centroids_history, partition_matrix = fuzzy_clustering.fuzzy(X, no_of_clusters)
        
        # Plotting
        plot_data_util(X, centroids, centroids_history, no_of_clusters)
        
        # Examine Cluster Validity with statistical tests
        initial_gamma, list_of_gammas, result = internal_criteria.internal_validity(X, no_of_clusters)
        initial_indices, list_of_indices, result_list = external_criteria.external_validity(X, no_of_clusters, y)
        
        # Histogram of gammas from internal criteria 
        hist_gamma_internal_criteria(initial_gamma, list_of_gammas, result)
        hist_gamma_external_criteria(initial_indices, list_of_indices, result_list)
        
        plt.show()
    
    @unittest.skip("no")
    def testCircles(self):
        no_of_clusters = 2
        
        # Create the dataset
        X, y = make_circles(n_samples=300, shuffle = True, noise = 0.05, factor = 0.5, random_state = 10)
        
        # Run the clustering Algorithm
        X, centroids, ita, centroids_history, partition_matrix = fuzzy_clustering.fuzzy(X, 2)
        
        # Plotting
        plot_data_util(X, centroids, centroids_history, no_of_clusters)
        
        # Examine Cluster Validity with statistical tests
        initial_gamma, list_of_gammas, result = internal_criteria.internal_validity(X, no_of_clusters)
        initial_indices, list_of_indices, result_list = external_criteria.external_validity(X, no_of_clusters, y)
        
        # Histogram of gammas from internal and external criteria 
        hist_gamma_internal_criteria(initial_gamma, list_of_gammas, result)
        hist_gamma_external_criteria(initial_indices, list_of_indices, result_list)
        
        plt.show()
        
    @unittest.skip("no")
    def testMoons(self):
        no_of_clusters = 2
        
        # Create the dataset
        X, y = make_moons(n_samples=300, shuffle = True, noise = 0.1, random_state = 10)
        
        # Run the clustering algorithm
        X, centroids, ita, centroids_history, partition_matrix = fuzzy_clustering.fuzzy(X, 2)
        
        # Plotting
        plot_data_util(X, centroids, centroids_history, no_of_clusters)
        
        # Examine Cluster Validity with statistical tests
        initial_gamma, list_of_gammas, result = internal_criteria.internal_validity(X, no_of_clusters)
        initial_indices, list_of_indices, result_list = external_criteria.external_validity(X, no_of_clusters, y)
        
        # Histogram of gammas from internal and external criteria 
        hist_gamma_internal_criteria(initial_gamma, list_of_gammas, result)
        hist_gamma_external_criteria(initial_indices, list_of_indices, result_list)
        
        plt.show()
    
    
    ################################################## Relative Criteria Clustering #########################
    
    #@unittest.skip('no')
    def testRelativeBlobs(self):
        
        no_of_clusters= 3
        
        # Create the dataset
        X, y = make_blobs(n_samples=10, centers= no_of_clusters, n_features=2,random_state=11)
        
        no_of_clusters_list = [i for i in range(2, 11)]
        values_of_q = [1.25, 2, 3, 4, 5]
        PC = np.zeros((len(no_of_clusters_list), len(values_of_q)))
        PE = np.zeros((len(no_of_clusters_list), len(values_of_q)))
         
        for i, total_clusters in tqdm(enumerate(no_of_clusters_list)): # no_of_clusters
            for j, q_value in enumerate(values_of_q): #edw vazw to q
                # When X returns it has one more column that needs to be erased
                X_, centroids, ita, centroids_history, partition_matrix = fuzzy_clustering.fuzzy(X, total_clusters, q = q_value)
                
                # Plotting
                #plot_data_util(X_, centroids, centroids_history, total_clusters)
                #plt.show()
                
                # Calculate index
                PC[i, j] = np.round(1/len(X) * np.sum(np.power(partition_matrix, 2)), 5)
                PE[i, j] = - 1/len(X) * np.sum(partition_matrix * np.log(partition_matrix))
                
                #################################
                #Xie Beni from here
                total_variation = 0.
                for k, centroid in enumerate(centroids):
                    temp = X - centroid
                    distances = np.sum(np.power(temp, 2), axis = 1).reshape(len(X), 1)
                    # alternative way
                    # distances = np.diagonal(np.dot(temp, temp.T)).reshape(len(X), 1)
                    cluster_variation = np.sum(np.power(partition_matrix[:, [k]], 2) * distances) # 2 here is the q value
                    total_variation += cluster_variation
                
                min_distance = max_integer
                for k, centroid1 in enumerate(centroids):
                    for l, centroid2 in enumerate(centroids):
                        if k != l:
                            temp = centroid1 - centroid2
                            distance = np.sum(np.power(temp, 2)) # it will always be 1 x 1, euclidean distance without the root
                            if min_distance > distance:
                                min_distance = distance 
                
                Xie_Beni = total_variation/(min_distance * len(X))
                print('lala')
                ####################################
        
        for j in range(len(values_of_q)):
            plt.plot(no_of_clusters_list, PC[:, j], '--')
            plt.show()
        
        for j in range(len(values_of_q)):
            plt.plot(no_of_clusters_list, PE[:, j])
            plt.show()
        
    
    
    
    
    
    
    
def plot_data_util(X, centroids, centroids_history ,no_of_clusters):
    np.random.seed(seed = None)
    clusters = np.unique(X[:, 2])
    
    # Plot Initial Data
    f, initDataPlot = plt.subplots(2, sharex=True,  figsize = (12,8))
    f.canvas.set_window_title('Unclustered and Clustered Data')
    initDataPlot[0].scatter(X[:, 0], X[:, 1])
    initDataPlot[0].set_title('Initial Data')
    initDataPlot[0].set_xlabel('Feature 1')
    initDataPlot[0].set_ylabel('Feature 2')
    
    # Plot data after clustering
    for i, cluster in enumerate(clusters):
        initDataPlot[1].scatter(X[ X[:,2] == cluster, 0], X[ X[:, 2] == cluster, 1], c=np.random.rand(3,1), s = 30)
    
    # Plots the centroids history
    colors= ['k', 'b', 'g', 'y', 'm', 'c']
    for alpha_counter, i in enumerate(range(0, len(centroids_history),  no_of_clusters)):
        for j in range(i, i + no_of_clusters):
            initDataPlot[1].plot(centroids_history[j, 0], centroids_history[j, 1], c = colors[j % len(colors)], marker = 'x', mew =  1, ms = 15, alpha = 0.2 + alpha_counter * 0.8/(len(centroids_history)/no_of_clusters))
            
    # Plots the centroids
    for i, c in enumerate(centroids):
            initDataPlot[1].plot(centroids[i, 0], centroids[i, 1], c = 'r', marker = 'x', mew=2, ms = 10)
    
    initDataPlot[1].set_title('Clustered Data')
    initDataPlot[1].set_xlabel('Feature 1')
    initDataPlot[1].set_ylabel('Feature 2')
    plt.tight_layout()
   

def hist_gamma_internal_criteria(initial_gamma, list_of_gammas, result):
    f, ax = plt.subplots(figsize = (12,8))
    f.canvas.set_window_title('Internal Criteria')
    n, bins, patches = plt.hist(list_of_gammas, bins = 30, color = 'g')
    ax.hist(initial_gamma, bins = 50, color = 'r')
    
    #bincenters = 0.5*(bins[1:]+bins[:-1])
    #y = norm.pdf(bincenters, np.mean(list_of_gammas), np.std(list_of_gammas))
    #ax.plot(bincenters, y, 'r--', linewidth=1)
    
    ax.set_title(result)
    #f.suptitle(result)
    ax.set_xlabel('Hubert\'s Gamma Values')
    ax.set_ylabel('Probability')
    
    plt.tight_layout()



def hist_gamma_external_criteria(initial_indices, list_of_indices, result_list):

    # row and column sharing
    f,  ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (12,8))
    f.canvas.set_window_title('External Criteria')
    figure_subplots = (ax1, ax2, ax3, ax4)
    names_of_indices = ['Rand statistic', 'Jaccard coefficient', 'Fowlkes and Mallows', 'Hubert\'s Gamma']
    for i in range(len(result_list)):
        n, bins, patches = figure_subplots[i].hist(list_of_indices[i, :], bins = 10, color = 'g')
        figure_subplots[i].hist(initial_indices[i], bins = 50, color = 'r')
        figure_subplots[i].set_title(result_list[i], fontsize=8, wrap = True, ha = 'center')
        figure_subplots[i].set_xlabel(names_of_indices[i])
        figure_subplots[i].set_ylabel('Probability')
        
        # Fit the normal distribution to the data
        #bincenters = 0.5*(bins[1:]+bins[:-1])
        #y = norm.pdf(bincenters, np.mean(list_of_indices[i, :]), np.std(list_of_indices[i, :]))
        #figure_subplots[i].plot(bincenters, y, 'r--', linewidth=1)
    
    #ax1.set_title('External indices')
    plt.tight_layout()
 




if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()