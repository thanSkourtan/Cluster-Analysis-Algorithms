
from cost_function_optimization import *
import numpy as np
import matplotlib.pyplot as plt

def external_indices(data, external_data_info):
    
    # Construct the four value table
    
    N= len(data)
    m = len(data[0])
    pair_matrix = np.zeros((N, N))
    X = np.zeros((N, N))
    Y = np.zeros((N, N))
    # 11: same C, same P -> SS
    # 22: different C, different P -> DD
    # 12: same C, different P -> SD
    # 21: different C, same P -> DS
    # The table is constructed in two phases, first for C, then for P
    
    for i in range(N): #for every vector, build its line
        C_same_cluster_indices = np.where(data[:, m - 1] == data[i, m - 1])
        P_same_cluster_indices = np.where(external_data_info  == external_data_info[i])
        
        # Build the arrays X, Y, to be used in the calculation of Hubert's Gamma
        X[i, C_same_cluster_indices[0]] = X[C_same_cluster_indices[0], i] = 1
        Y[i, P_same_cluster_indices[0]] = Y[P_same_cluster_indices[0], i] = 1
        
        # Convert to sets so as to use set operations
        C_set = set(C_same_cluster_indices[0])
        P_set = set(P_same_cluster_indices[0])
        complete_set = set([k for k in range(N)])
        
        
        set_11 = (C_set & P_set) - set([i]) # diagonal should stay 0
        set_22 = complete_set - C_set - P_set
        set_12 = C_set - P_set
        set_21 = P_set - C_set
        
        pair_matrix[i , list(set_11)] = 11
        pair_matrix[i , list(set_22)] = 22
        pair_matrix[i , list(set_12)] = 12
        pair_matrix[i , list(set_21)] = 21
        
    # Count occurrences. They are the total/ 2 except for 11 where the diagonal must be deleted
    a = len(np.where(pair_matrix == 11)[0])/2
    b = len(np.where(pair_matrix == 12)[0])/2
    c = len(np.where(pair_matrix == 21)[0])/2
    d = len(np.where(pair_matrix == 22)[0])/2
        
    # Calculate all indices
    M = N*(N - 1)/2
    
    rand_statistic = (a + d)/M
    jaccard_coefficient = a/(a + b + c)
    fowlkes_and_mallows = np.sqrt(a/(a + b) * a/(a + c))
    
    total_sum = 0.
    for i in range(N):
        total_sum += np.sum(X[i, i + 1:] * Y[i, i + 1:])
    gamma =  total_sum / M
    
    return rand_statistic, jaccard_coefficient, fowlkes_and_mallows, gamma

def monte_carlo(data, no_of_clusters, external_data_info):
    N = len(data)
    m = len(data[0])
    
    # Monte Carlo simulation - create the datasets (random position hypothesis)
    list_of_indices = np.zeros((4, 0)) #cause we have 4 indices
    for j in range(100):
        random_data = np.empty((N, 0))
        #debug
        print('here')
        
        for i  in range(m - 1):
            max_value = np.amax(data[:, i])
            min_value = np.min(data[:, i])
            temp = (max_value - min_value) * np.random.random(size = (N, 1)) + min_value
            random_data = np.concatenate((random_data, temp), axis = 1)
            
        X, centroids, ita, centroids_history, partition_matrix = fuzzy_clustering.fuzzy(random_data, no_of_clusters)
        
        temp = np.array(external_indices(X, external_data_info)).reshape(4, 1)
        list_of_indices= np.concatenate((list_of_indices, temp), axis = 1)
    
    return list_of_indices


def external_validity(data, no_of_clusters, external_data_info):
    initial_indices = external_indices(data, external_data_info)
    list_of_indices = monte_carlo(data, no_of_clusters, external_data_info)
    
    return initial_indices, list_of_indices

    
    
    
    
    