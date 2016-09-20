from scipy.stats import norm
from cost_function_optimization import fuzzy_clustering, possibilistic_clustering, kmeans_clustering
from sequential import BSAS, TTSS
from graph_theory import MST, MST_Eld_Heg_Var
import numpy as np
from tqdm import tqdm


euclidean_distance = lambda data, point: np.sqrt(np.sum(np.power(data - point, 2), axis = 1).reshape((len(data), 1)))

def gamma(data):
    ''' Calculates the Hubert's Gamma Statistic for the proximity matrix P and Y, where Y (i,j) = 0 
        if i, j are in the same cluster, 1 otherwise. These inputs are fixed for the internal criteria case
        so they are integrated into this function.
    
    Parameters:
        data((m x n) 2-d numpy array): a data set of m instances and n features
    
    Returns:
        g(float): the gamma index for P, Y
        
    Reference: Pattern Recognition, S. Theodoridis, K. Koutroumbas
    
    '''
    N = len(data)
    m = len(data[0]) - 1
    
    # Construct the proximity matrix P. This always takes a lot of time.
    P = np.empty((N, N)) 
    for i, point in enumerate(data):
        P[:, [i]] =  euclidean_distance(data[:, :m],point[:m])
    
    # Construct the matrix Y
    Y = np.zeros((N, N))
    for i, _ in enumerate(data):
        same_cluster_indices = np.where(data[:, m] == data[i, m])[0]
        Y[i, same_cluster_indices] = 1
    
    # Calculate the Hubert's Gamma Statistic    
    M = N * (N - 1) / 2
    total_sum = 0.
    for i in range(N):
        total_sum += np.sum(P[i, i + 1:] * Y[i, i + 1:])
    g =  total_sum / M
    
    return g
    


def monte_carlo(data, no_of_clusters, algorithm):
    ''' Creates 100 (could be set as argument) sampling distributions of uniformingly distributed data and
        calls the appropriate functions in order to cluster each distribution and calculate its Gamma statistic.
        
    Parameters:
        data((m x n) 2-d numpy array): a data set of m instances and n features
        no_of_clusters(integer): the number of clusters
    
    Returns:
        list_of_gammas(list): the Gamma statistics of all the monte carlo sample distributions
        
    '''
    N = len(data)
    m = len(data[0]) - 1
    
    # Monte Carlo simulation - create the datasets (random position hypothesis)
    list_of_gammas = []
    #pbar = tqdm(range(100))
    #pbar.set_description('Monte carlo sim. - internal indices')
    j = 0
    while j < 100:
        random_data = np.empty((N, m))
        
        for i  in range(m):
            max_value = np.amax(data[:, i])
            min_value = np.min(data[:, i])
            temp = (max_value - min_value) * np.random.random(size = (N, 1)) + min_value
            random_data[:, [i]] = temp
        
        if algorithm == fuzzy_clustering.fuzzy:
            X, centroids, ita, centroids_history, partition_matrix = algorithm(random_data, no_of_clusters)
        elif algorithm == possibilistic_clustering.possibilistic:
            X_, centroids, ita, centroids_history, partition_matrix = fuzzy_clustering.fuzzy(random_data, no_of_clusters)
            X, centroids, centroids_history, typicality_matrix = algorithm(random_data, no_of_clusters, ita, centroids_initial = centroids)
        elif algorithm == kmeans_clustering.kmeans:
            X, centroids, centroids_history = algorithm(random_data, no_of_clusters)
        elif algorithm == BSAS.basic_sequential_scheme:
            X, centroids, no_of_clusters = algorithm(random_data)
            if(X is None):
                continue # Being able to rerun this loop is the reason we use a while instead of a for loop
        elif algorithm == TTSS.two_threshold_sequential_scheme:
            X, centroids, no_of_clusters = algorithm(random_data)
        elif algorithm == MST.minimum_spanning_tree:
            X, no_of_clusters = algorithm(random_data)
        elif algorithm == MST_Eld_Heg_Var.minimum_spanning_tree_variation:
            X, no_of_clusters = algorithm(random_data)

        list_of_gammas.append(gamma(X))
        print(j)
        j += 1
    
    return list_of_gammas
    

def significance_calc(initial_gamma, list_of_gammas):
    ''' Calculates z-statistic for initial_gamma with regards to the normal distribution of list_of_gammas
        the p_value of the z-statistic and based on the results accepts or rejects the null hypothesis of 
        randomness.
        
    Parameters:
        initial_gamma(float): the Gamma statistic of the clustering under consideration
        list_of_gammas(list): the Gamma statistics of all the monte carlo sample distributions
    
    Returns:
        result(string): a string containing the result of the function's computations
        
    '''
    z_statistic = (initial_gamma - np.mean(list_of_gammas))/np.std(list_of_gammas)
    # Two tailed test
    if z_statistic <= 0:
        p_value1 = norm.cdf(z_statistic)
        p_value2 = 1 - norm.cdf(-z_statistic)
    else:
        p_value1 = 1 - norm.cdf(z_statistic)
        p_value2 = norm.cdf(-z_statistic)
        
    if p_value1 + p_value2 < 0.05:
        result = 'The null hypothesis was rejected for significance level 0.05, with p_value = {:f}'.format(p_value1 + p_value2)
    else:
        result = 'The null hypothesis was accepted for significance level 0.05, with p_value = {:f}'.format(p_value1 + p_value2)
    return result


def internal_validity(data, no_of_clusters, algorithm):
    ''' A function that wraps the rest of the functions of this module and calls them in the 
        appropriate order. It could be defined as the only public function of the module. 
        
    Parameters:
        data((m x n) 2-d numpy array): a data set of m instances and n features
        no_of_clusters(integer): the number of clusters
    
    Returns:
        initial_gamma(float): the Gamma statistic of the clustering under consideration
        list_of_gammas(list): the Gamma statistics of all the monte carlo sample distributions
        result(string): a string containing the result of the function's computations
        
    '''
    initial_gamma = gamma(data)
    list_of_gammas = monte_carlo(data, no_of_clusters, algorithm)
    result = significance_calc(initial_gamma, list_of_gammas)
    
    return initial_gamma, list_of_gammas, result
    
    
    
    
    
    