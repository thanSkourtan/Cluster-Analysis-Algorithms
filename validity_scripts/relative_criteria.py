import numpy as np
from tqdm import tqdm
from cost_function_optimization import fuzzy_clustering, possibilistic_clustering, kmeans_clustering
from sys import maxsize as max_integer
import matplotlib.pyplot as plt
from utility.plotting_functions import *

euclidean_distance = lambda data, point: np.sqrt(np.sum(np.power(data - point, 2), axis = 1).reshape((len(data), 1)))


def relative_validity_hard(X, no_of_clusters):
    # Initialization
    no_of_clusters_list = [i for i in range(2, 11)]
    
    DI = np.zeros(len(no_of_clusters_list))
    DB = np.zeros(len(no_of_clusters_list))
    SI = np.zeros(len(no_of_clusters_list))
    
    for i, total_clusters in tqdm(enumerate(no_of_clusters_list)): # no_of_clusters
        X_, centroids, centroids_history = kmeans_clustering.kmeans(X, no_of_clusters)
        
        DI[i] = Dunn_index(X_)
        DB[i] = Davies_Bouldin(X_, centroids)
        SI[i] = silhouette_index(X_)
        
        # Print just one clustering effort, the correct one in order to compare it with the indices' signals
        if total_clusters == no_of_clusters:
            plot_data(X_, centroids, total_clusters, centroids_history)
            
    
    return no_of_clusters_list, DI, DB, SI


def relative_validity_fuzzy(X, no_of_clusters):
    ''' Constructs the framework into which successive executions of the 
        algorithm take place
        
        Parameters:
        X((m x n) 2-d numpy array): a data set of m instances and n features
        no_of_clusters: the number of clusters
        
        Returns:
        no_of_clusters_list: the different number of clusters tried
        values_of_q: the different values of q that were tried.
        PC, PE, XB, FS : the arrays holding the values of the four indices
    '''
    # Initialization
    no_of_clusters_list = [i for i in range(2, 11)]
    values_of_q = [1.25, 1.5, 2, 2.5, 3, 3.5, 5]
    
    # Initialize arrays to hold the indices. We use separate arrays for easier modification of the code if needed.
    # If we wanted to use one array then this would be a 3 - dimensional array.
    PC = np.zeros((len(no_of_clusters_list), len(values_of_q)))
    PE = np.zeros((len(no_of_clusters_list), len(values_of_q)))
    XB = np.zeros((len(no_of_clusters_list), len(values_of_q)))
    FS = np.zeros((len(no_of_clusters_list), len(values_of_q)))
    
    for i, total_clusters in tqdm(enumerate(no_of_clusters_list)): # no_of_clusters
        # IMPORTANT: The centroids must remain the same for every run of the algorithm with the same no_of_clusters
        centroids_initial = np.random.choice(np.arange(np.min(X), np.max(X), 0.1), size = (total_clusters, len(X[0])), replace = False)
        
        for j, q_value in enumerate(values_of_q): #edw vazw to q
            
            # When X returns it has one more column that needs to be erased
            X_, centroids, ita, centroids_history, partition_matrix = fuzzy_clustering.fuzzy(X, total_clusters, centroids_initial, q = q_value)
                
            # Calculate indices
            PC[i, j] = partition_coefficient(X, partition_matrix)
            PE[i, j] = partition_entropy(X, partition_matrix)
            XB[i, j] = Xie_Beni(X, centroids, partition_matrix)
            FS[i, j] = fukuyama_sugeno(X, centroids, partition_matrix, q = 2)
             
            # Print just one clustering effort, the correct one in order to compare it with the indices' signals
            if q_value == 1.25 and total_clusters == no_of_clusters:
                plot_data(X_, centroids, total_clusters, centroids_history)
                
            
    return no_of_clusters_list, values_of_q, PC, PE, XB, FS
    
        
    


# Lambda functions in order to calculate the same name indices
partition_coefficient = lambda X, partition_matrix: np.round(1/len(X) * np.sum(np.power(partition_matrix, 2)), 5)
partition_entropy = lambda X, partition_matrix: - 1/len(X) * np.sum(partition_matrix * np.log(partition_matrix)) 



def Dunn_index(X):
    

    N = len(X)
    m = len(X[0])
    clusters = np.unique(X[:, m - 1])
    # The two basic structures of the index. The distance_between_clusters is an upper triangular matrix
    #distance_between_clusters = np.zeros((len(clusters), len(clusters)))
    #cluster_diameter = np.zeros((len(clusters)))
    min_cluster_distance = max_integer
    max_cluster_diameter = -max_integer - 1
    
    # Construct the dissimilarity matrix
    dissimilarity_matrix = np.empty((N, N)) 
    for j, point in enumerate(X):
        dissimilarity_matrix[:, [j]] = euclidean_distance(X, point)
    
    for i, cluster1 in enumerate(clusters):
        # Calculate the diameter of each cluster
        first_cluster_indices = np.where(X[:, m - 1] == cluster1)[0]
        temp = np.max(dissimilarity_matrix[first_cluster_indices.reshape(len(first_cluster_indices), 1), first_cluster_indices])
        if max_cluster_diameter < temp:
            max_cluster_diameter = temp
        
        for j, cluster2 in enumerate(clusters[(i+1):], start = i + 1):
            # Calculate the distances between the clusters
            second_cluster_indices = np.where(X[:, m - 1] == cluster2)[0]
            # The reshape creates a n x 1 2-d array which is very important for the indexing of the dissimilarity matrix
            temp = np.min(dissimilarity_matrix[first_cluster_indices.reshape(len(first_cluster_indices), 1), second_cluster_indices])
            if min_cluster_distance > temp:
                min_cluster_distance = temp
    
    # Dunn index is the minimum distance between clusters divided by the maximum diameter
    return min_cluster_distance/max_cluster_diameter


def Davies_Bouldin(X, centroids):
    
    # If a centroids has not been used, the index is implemented in such a way that it is skipped
    #pote den xrisimopoioume tous centroids monous tous, panta pairnoume mono ta used clustesr
    # In Dunn index the distance between clusters is the distance between the closest vectors of the clusters
    # In Davies Bouldin the same distance is the distance between the centroids.
    
    N = len(X)
    m = len(X[0])
    
    clusters = np.unique(X[:, m - 1])
    # Casting the clusters array to int as we are going to use it later for indexing
    clusters = clusters.astype(int)
    
    # Create a 1-D matrix to hold the values of each cluster's dispersion
    cluster_dispersion = np.zeros((len(clusters)))
    # Create a dissimilarity matrix to hold the distances between the clusters' centroids
    cluster_distances = np.zeros((len(clusters), len(clusters)))
    
    # Calculate dispersion values and clusters' distances in one loop
    for i, cluster in enumerate(clusters): 
        temp = np.sum(np.power(X[np.where(X[:, m - 1] == cluster)[0], :(m - 1)] - centroids[cluster], 2))
        cluster_dispersion[i] = np.sqrt(1/N * temp)
        # Calculate clusters' distances
        cluster_distances[i, :] = euclidean_distance(centroids[clusters, :], centroids[cluster]).reshape(1, len(clusters))
    
    # Create a matrix to hold the similarity indices between clusters
    R = np.zeros((len(clusters), len(clusters)))
    for i, cluster1 in enumerate(clusters):
        for j, cluster2 in enumerate(clusters[(i+1):], start = i + 1):
            R[i, j] =  (cluster_dispersion[i] + cluster_dispersion[j]) / cluster_distances[i, j] 
    
    DB = np.average(np.amax(R, axis = 1))
    
    return DB

        

def silhouette_index(X):
    
    N= len(X)
    m = len(X[0])
    clusters = np.unique(X[:, m - 1])
    
    # Construct the dissimilarity matrix
    dissimilarity_matrix = np.empty((N, N)) 
    for j, point in enumerate(X):
        dissimilarity_matrix[:, [j]] = euclidean_distance(X, point)
    
    # a: average_distance_in_same_cluster. Average distance only for the vectors belonging to the same clusters
    a = np.zeros((N))
    
    # Calculates the silhouettes of all clusters as the average silhouettes of their vectors
    # Calculates 
    for i, cluster in enumerate(clusters):
        cluster_indices = np.where(X[:, m - 1] == cluster)[0]
        # Number of vectors in the cluster
        n = len(cluster_indices)
        cluster_dissimmilarity_matrix = dissimilarity_matrix[cluster_indices.reshape(n, 1), cluster_indices]
            
        for j, vector_index in enumerate(cluster_indices):
            a[vector_index] = np.sum(cluster_dissimmilarity_matrix[j, :], axis = 0)/(n - 1)
        
    #  b: average_distance_in_closest_cluster. Average distance for vectors belonging to the closest cluster
    b = np.zeros((N))
    b.fill(max_integer)
    
    # Calculates
    for i, cluster1 in enumerate(clusters):
        cluster_indices1 = np.where(X[:, m - 1] == cluster1)[0]
        # Number of vectors in the cluster
        n = len(cluster_indices1)
        for j, cluster2 in enumerate(clusters):
            if cluster1 != cluster2:
                cluster_indices2 = np.where(X[:, m - 1] == cluster2)[0]
                k = len(cluster_indices2)
                different_cluster_dissimmilarity_matrix = dissimilarity_matrix[cluster_indices1.reshape(n, 1), cluster_indices2]
                
                # Here we divide with k instead of k - 1, because the same vector will never be at the same cluster
                for j, vector_index in enumerate(cluster_indices1):
                    if b[vector_index] > np.sum(different_cluster_dissimmilarity_matrix[j, :], axis = 0)/k:
                        b[vector_index] = np.sum(different_cluster_dissimmilarity_matrix[j, :], axis = 0)/k
        
                
    # Calculates the silhouette width of every vector
    vector_silhouette_width = (b - a)/np.amax((b,a), axis = 0)
    
    # Calculates the silhouette width of every cluster
    cluster_silhouette_width = np.zeros(len(clusters))
    for i, cluster1 in enumerate(clusters):
        cluster_indices = np.where(X[:, m - 1] == cluster1)[0]
        cluster_silhouette_width[i] = np.average(vector_silhouette_width[cluster_indices])
    
    # Calculates the global silhouette index
    global_silhouette_index = np.average(cluster_silhouette_width)
    
    return global_silhouette_index


def Xie_Beni(X, centroids, partition_matrix):
    ''' Calculates the Xie Beni index.
    
    Parameters:
        X((m x n) 2-d numpy array): a data set of m instances and n features
        centroids: the value of the centroids after running a clustering algorihtm on the data set
        partition_matrix: the partition matrix
    
    Returns:
        Xie_Beni(float): the value of the Xie Beni index
        
    Reference: Pattern Recognition, S. Theodoridis, K. Koutroumbas
    '''
    total_variation = 0.
    for k, centroid in enumerate(centroids):
        temp = X - centroid
        distances = np.sum(np.power(temp, 2), axis = 1).reshape(len(X), 1)
        # alternative way
        # distances = np.diagonal(np.dot(temp, temp.T)).reshape(len(X), 1) na dw kai trace
        cluster_variation = np.sum(np.power(partition_matrix[:, [k]], 2) * distances) # 2 here is the q value
        total_variation += cluster_variation
                
    min_distance = max_integer
    for k, centroid1 in enumerate(centroids):
        for l, centroid2 in enumerate(centroids):
            if k < l:
                temp = centroid1 - centroid2
                distance = np.sum(np.power(temp, 2)) # it will always be 1 x 1, euclidean distance without the root
                if min_distance > distance:
                    min_distance = distance 
                
    Xie_Beni = total_variation/(min_distance * len(X))
    
    return Xie_Beni


def fukuyama_sugeno(X, centroids, partition_matrix, q = 2):
    ''' Calculates the fukuyama sugeno index
    '''
    w = np.mean(X, axis = 0)
    total_sum = 0.
    for k, centroid in enumerate(centroids):
        term1 = X - centroid
        distances1 = np.sum(np.power(term1, 2), axis = 1).reshape(len(X), 1) 
        
        term2 = centroid - w
        distances2 = np.sum(np.power(term2, 2))
        
        temp = distances1 - distances2
        total_sum += np.sum(np.power(partition_matrix[:, [k]], q) * temp)
    
    return total_sum

















