import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt

euclidean_distance = lambda data, point: np.sqrt(np.sum(np.power(data - point, 2), axis = 1).reshape((len(data), 1)))

def two_threshold_sequential_scheme(data):
    ''' An implementation of the two threshold sequential scheme clustering algorithm.
    
    Parameters:
        data((m x n) 2-d numpy array): a data set of m instances and n features
    
    Returns:
        clustered_data((m x (n + 1)) 2-d numpy array): the data set with one more column that contains the vector's cluster
        centroids((k x n)2-d numpy array): contains the k = no_of_clusters centroids with n features
        no_of_clusters(integer): the final number of clusters created
    
    '''
    N = len(data)
    
    # Automatically calculating threshold by peaks and valleys technique
    threshold1, threshold2 = thresholding_TTSS(data)
    
    # Creating an array to keep track of the vectors that have been processed and assigned to a cluster
    processed = np.zeros((N))
    
    # We keep two copies of the data matrix, one with the cluster column
    clusters = np.zeros((N, 1))
    clusters.fill(-1)
    clustered_data = np.concatenate((data, clusters), axis = 1)
    m = len(clustered_data[0])
    
    # Assign the first point to this cluster
    cluster_index = 0
    clustered_data[0, m - 1] = 0
    centroids = np.array([data[0]])
    processed[0] = 1
    
    flag_change = False
    
    while(not np.all(processed)): # when all becomes 1 then it is false and exits

        flag_change = False
        for i, vector in enumerate(data[1:], start = 1):

            if processed[i] == 0:
                distance_from_centroids = euclidean_distance(centroids, vector)
                nearest_cluster_distance = np.min(distance_from_centroids)
                nearest_cluster = np.argmin(distance_from_centroids)
                if nearest_cluster_distance < threshold1:
                    clustered_data[i, m - 1] = nearest_cluster
                    # Update Centroids
                    vectors_in_cluster = len(np.where(clustered_data[:, m - 1] == nearest_cluster)[0])
                    centroids[nearest_cluster] = ((vectors_in_cluster - 1) * centroids[nearest_cluster] + vector) /vectors_in_cluster
                    processed[i] = 1
                    flag_change = True
                    
                elif nearest_cluster_distance > threshold2:
                    cluster_index += 1
                    clustered_data[i, m - 1] = cluster_index
                    centroids = np.concatenate((centroids, [data[i]]), axis = 0)
                    processed[i] = 1
                    flag_change = True
        
        # If no change took place during a pass from the data        
        if flag_change == False:
            current_vector = np.nonzero(processed == 0)[0][0]
            cluster_index += 1
            clustered_data[current_vector, m - 1] = cluster_index #create a new cluster for the first nonzero element
            centroids = np.concatenate((centroids, [data[current_vector]]), axis = 0)
            processed[current_vector] = 1
            
    return clustered_data, centroids, cluster_index + 1



def thresholding_TTSS(data):
    ''' A function to calculate the values of the thresholds

    Parameters:
        data((m x n) 2-d numpy array): a data set of m instances and n features
    
    Returns:
        deepest_valley1(float): the height of the histogram at the point of the first deepest valley
                               between the three highest peeks. It is actually the threshold 1 value
        deepest_valley2(float): the height of the histogram at the point of the second deepest valley
                               between the three highest peeks. It is actually the threshold 2 value
    
    '''
    # Construct the dissimilarity matrix
    N = len(data)
    dissimilarity_matrix = np.empty((N, 0)) 
    for point in data:
        dissimilarity_matrix = np.concatenate((dissimilarity_matrix, euclidean_distance(data,point)), axis=1)
    
    distances = np.zeros((N * (N - 1)/2)) #number of pairs
    
    k = 0
    for i, row in enumerate(dissimilarity_matrix):
        temp_data = row[(i + 1):]
        distances[k: k + len(temp_data)] = temp_data
        k += len(temp_data)
    
    n, bins  = np.histogram(distances, bins = 50) #calculating, not plotting
    # n, bins, patches = plt.hist(distances, bins = 50, color = 'g')

    # Peak and valley seeking    
    all_peaks_indices = argrelextrema(n, np.greater)[0]
    all_peaks_values = n[all_peaks_indices]
    sorted_list_of_peaks_indices = [index for value, index in sorted(zip(all_peaks_values, all_peaks_indices))]
    three_largest_peaks = sorted_list_of_peaks_indices[-3:]
    temp = sorted(three_largest_peaks)
    
    # The first if statement applies to monte carlo simulations where the data do not have structure, 
    # so there might be only one peak
    if len(temp) == 1:#if there is only one peak
        two_deepest_valley_bin = (temp[0], temp[0] + 0.01) # in this rare case that takes place only in monte carlo simulations put these values
    elif len(temp) == 2:
        two_deepest_valley_bin = (temp[0], temp[1])
    else:
        two_deepest_valley_bin = argrelextrema(n[temp[0]:temp[2] + 1], np.less_equal)[0] + temp[0]

    deepest_valley1 = bins[two_deepest_valley_bin[0]]
    deepest_valley2 = bins[two_deepest_valley_bin[1]]

    return deepest_valley1, deepest_valley2




