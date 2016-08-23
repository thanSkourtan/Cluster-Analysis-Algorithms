
from cost_function_optimization import *
import numpy as np
import matplotlib.pyplot as plt

euclidean_distance = lambda data, point: np.sqrt(np.sum(np.power(data - point, 2), axis = 1).reshape((len(data), 1)))

def gamma(data, no_of_clusters):
    N = len(data)
    m = len(data[0])
    
    # Construct the proximity matrix P
    P = np.empty((N, 0)) 
    for point in data:
        P = np.concatenate((P, euclidean_distance(data,point)), axis=1)
    
    
    # Calculate the Hubert's Gamma Statistic
    Y = np.zeros((N, N))
    
    for i, d in enumerate(data):
        same_cluster_indices = np.where(data[:, m - 1] == data[i, m - 1])
        Y[i, same_cluster_indices[0]] = Y[same_cluster_indices[0], i] = 1
        
    M = N * (N - 1) / 2
    total_sum = 0.
    for i in range(N):
        total_sum += np.sum(P[i, i + 1:] * Y[i, i + 1:])
    g =  total_sum / M
    
    return g
    


def monte_carlo(data, no_of_clusters):
    N = len(data)
    m = len(data[0])
    
    # Monte Carlo simulation - create the datasets (random position hypothesis)
    list_of_gammas = []
    for j in range(100):
        random_data = np.empty((N, 0))
        #debug
        print('here')
        
        for i  in range(m - 1):
            max_value = np.amax(data[:, i])
            min_value = np.min(data[:, i])
            temp = (max_value - min_value) * np.random.random(size = (N, 1)) + min_value
            random_data = np.concatenate((random_data, temp), axis = 1)
            
        X, centroids, ita, centroids_history = fuzzy_clustering.fuzzy(random_data, no_of_clusters)
        
        list_of_gammas.append(gamma(X, no_of_clusters))
    
    return list_of_gammas
    
        

def internal_validity(data, no_of_clusters):
    initial_gamma = gamma(data, no_of_clusters)
    list_of_gammas = monte_carlo(data, no_of_clusters)
    
    z_statistic = (initial_gamma - np.mean(list_of_gammas))/(np.std(list_of_gammas)/ np.sqrt(100)) #because std is for samples
    
    #do more stuff
    
    return initial_gamma, list_of_gammas
    
    
    
    
    
    