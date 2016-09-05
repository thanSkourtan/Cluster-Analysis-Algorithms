import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt

euclidean_distance = lambda data, point: np.sqrt(np.sum(np.power(data - point, 2), axis = 1).reshape((len(data), 1)))

def two_threshold_sequential_scheme(data):
    
    N = len(data)
    threshold1, threshold2 = thresholding_TTSS(data)
    
    processed = np.zeros((N))
    
    clusters = np.zeros((N, 1))
    clusters.fill(-1)
    clustered_data = np.concatenate((data, clusters), axis = 1)
    
    m = len(clustered_data[0])
    
    cluster_index = 0
    # assign the first point to this cluster 
    clustered_data[0, m - 1] = 0
    # the centroid of this first cluster is the data[0] vector
    centroids = np.array([data[0]])
    processed[0] = 1
    
    '''
    cluster_index = -1
    centroids = np.empty((0, m - 1))
    '''
    
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
        
        #if no change happened during a pass from the data        
        if flag_change == False:
            current_vector = np.nonzero(processed == 0)[0][0]
            cluster_index += 1
            clustered_data[current_vector, m - 1] = cluster_index #create a new cluster for the first nonzero element
            centroids = np.concatenate((centroids, [data[current_vector]]), axis = 0)
            processed[current_vector] = 1
            
                        
        
    
    return clustered_data, centroids, cluster_index + 1



def thresholding_TTSS(data):
    ''' A function to calculate the value of the threshold
    '''
    #construct the dissimilarity matrix
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
    
    n, bins, patches = plt.hist(distances, bins = 50, color = 'g')
    
    # Peak and valley seeking
    
    all_peaks_indices = argrelextrema(n, np.greater)[0]
    all_peaks_values = n[all_peaks_indices]
    sorted_list_of_peaks_indices = [index for value, index in sorted(zip(all_peaks_values, all_peaks_indices))]
    
    if len(sorted_list_of_peaks_indices) < 2:
        print("The algorithm could not find three picks. Try BSAS algorithm instead.")
        return
    
    three_largest_peaks = sorted_list_of_peaks_indices[-3:]
    temp = sorted(three_largest_peaks)
    
    two_deepest_valley_bin = argrelextrema(n[temp[0]:temp[2]], np.less)[0] + temp[0]
    
    deepest_valley1 = bins[two_deepest_valley_bin[0]]
    deepest_valley2 = bins[two_deepest_valley_bin[1]]

    
    print(deepest_valley1)
    print(deepest_valley2)
    return deepest_valley1, deepest_valley2




