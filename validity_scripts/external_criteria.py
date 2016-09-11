from scipy.stats import norm
from cost_function_optimization import *
from sequential import *
import numpy as np
from tqdm import tqdm
from graph_theory import *


def external_indices(data, external_data_info):
    '''Calculates three indices (rand statistic, jaccard coefficient, Fowlkes and Mallows) based on a matrix P
       that shows the simmilarity between the clustering under consideration and an external clustering. Also
       calculates the Hubert's Gamma Statistic for matrices X and Y, where X (i,j) = 1 if i, j are in the same cluster
       in the clustering under consideration, 0 otherwise and  Y(i,j) = 1 if i, j are in the same cluster
       in the external clustering,  0 otherwise. 
       
    Parameters:
        data((m x n) 2-d numpy array): a data set of m instances and n features
        external_data_info(list): the external clustering results
    
    Returns:
        rand_statistic(float): the rand statistic
        jaccard_coefficient(float): the jaccard coefficient statistic 
        fowlkes_and_mallows(float): the Fowlkes and Mallows index
        gamma(float): the gamma index for X, Y
        
    Reference: Pattern Recognition, S. Theodoridis, K. Koutroumbas
    '''
    # Construct three matrices. X and Y to be used for Gamma Statistic and P for all other indices
    
    N= len(data)
    m = len(data[0])
    P = np.zeros((N, N))
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
        
        P[i , list(set_11)] = 11
        P[i , list(set_22)] = 22
        P[i , list(set_12)] = 12
        P[i , list(set_21)] = 21
        
    # Count occurrences. They are the total/ 2 
    a = len(np.where(P == 11)[0])/2
    b = len(np.where(P == 12)[0])/2
    c = len(np.where(P == 21)[0])/2
    d = len(np.where(P == 22)[0])/2
        
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

def monte_carlo(data, no_of_clusters, external_data_info, algorithm):
    ''' Creates 100 (could be set as argument) sampling distributions of uniformingly distributed data and
        calls the appropriate functions in order to cluster each distribution and calculate its Gamma statistic.
        
    Parameters:
        data((m x n) 2-d numpy array): a data set of m instances and n features
        no_of_clusters(integer): the number of clusters
        external_data_info(list): the external clustering results
    
    Returns:
        list_of_indices(list): the calculated indices of all the monte carlo sample distributions
        
    '''
    N = len(data)
    m = len(data[0])
    
    # Monte Carlo simulation - create the datasets (random position hypothesis)
    list_of_indices = np.zeros((4, 0)) #cause we have 4 indices
    pbar = tqdm(range(100))
    pbar.set_description('Monte carlo sim. - external indices')
    
    for _ in pbar:
        random_data = np.empty((N, 0))
        
        for i  in range(m - 1):
            max_value = np.amax(data[:, i])
            min_value = np.min(data[:, i])
            temp = (max_value - min_value) * np.random.random(size = (N, 1)) + min_value
            random_data = np.concatenate((random_data, temp), axis = 1)
            
        if algorithm == fuzzy_clustering.fuzzy:
            X, centroids, ita, centroids_history, partition_matrix = algorithm(random_data, no_of_clusters)
        elif algorithm == possibilistic_clustering.possibilistic:
            X_, centroids, ita, centroids_history, partition_matrix = fuzzy_clustering.fuzzy(random_data, no_of_clusters)
            X, centroids, centroids_history, typicality_matrix = algorithm(random_data, no_of_clusters, ita, centroids_initial = centroids)
        elif algorithm == kmeans_clustering.kmeans:
            X, centroids, centroids_history = algorithm(random_data, no_of_clusters)
        elif algorithm == BSAS.basic_sequential_scheme:
            X, centroids, no_of_clusters = algorithm(random_data)
        elif algorithm == TTSS.two_threshold_sequential_scheme:
            X, centroids, no_of_clusters = algorithm(random_data)
        elif algorithm == MST.minimum_spanning_tree:
            X, no_of_clusters = algorithm(random_data)
        elif algorithm == MST_Eld_Heg_Var.minimum_spanning_tree_variation:
            X, no_of_clusters = algorithm(random_data)



        temp = np.array(external_indices(X, external_data_info)).reshape(4, 1)
        list_of_indices= np.concatenate((list_of_indices, temp), axis = 1)
    
    return list_of_indices


def significance_calc(initial_indices, list_of_indices):
    ''' Calculates z-statistic for initial_indices with regards to the normal distribution of list_of_gammas
        the p_value of the z-statistic and based on the results accepts or rejects the null hypothesis of 
        randomness.
        
    Parameters:
        initial_indices(float): the initial indices of the clustering under consideration
        list_of_indices(list): the list of calculated indices of all the monte carlo sample distributions
        
    Returns:
        result(list): a list of strings containing the results of the function's computations
        
    '''
    no_of_indices = len(list_of_indices)
    result_list = [0] * no_of_indices #one result for each index
    for i in range(no_of_indices):
        z_statistic = (initial_indices[i] - np.mean(list_of_indices[i, :]))/np.std(list_of_indices[i, :])
        # Two tailed test
        if z_statistic <= 0:
            p_value1 = norm.cdf(z_statistic)
            p_value2 = 1 - norm.cdf(-z_statistic)
        else:
            p_value1 = 1 - norm.cdf(z_statistic)
            p_value2 = norm.cdf(-z_statistic)
        
        if p_value1 + p_value2 < 0.05:
            result_list[i] = 'Null hypothesis rejected for significance level 0.05, with p_value = {:f}'.format(p_value1 + p_value2)
        else:
            result_list[i] = 'Null hypothesis accepted for significance level 0.05, with p_value = {:f}'.format(p_value1 + p_value2)
    
    return result_list


def external_validity(data, no_of_clusters, external_data_info, algorithm):
    ''' A function that wraps the rest of the functions of this module and calls them in the 
        appropriate order. It could be defined as the only public function of the module. 
        
    Parameters:
        data((m x n) 2-d numpy array): a data set of m instances and n features
        no_of_clusters(integer): the number of clusters
        external_data_info(list): the external clustering results
    
    Returns:
        initial_indices(float): the initial indices of the clustering under consideration
        list_of_indices(list): the list of calculated indices of all the monte carlo sample distributions
        result_list(list): a list of strings containing the results of the function's computations
        
    '''
    initial_indices = external_indices(data, external_data_info)
    list_of_indices = monte_carlo(data, no_of_clusters, external_data_info, algorithm)
    result_list = significance_calc(initial_indices, list_of_indices)
    
    return initial_indices, list_of_indices, result_list

    
    
    
    
    