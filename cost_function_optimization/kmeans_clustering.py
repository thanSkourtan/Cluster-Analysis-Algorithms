
import numpy as np

euclidean_distance = lambda data, point: np.sqrt(np.sum(np.power(data - point, 2), axis = 1).reshape((len(data), 1)))

def kmeans(data, no_of_clusters, centroids_initial = None):
    ''' An implementation of the kmeans clustering algorithm.
    
    Parameters:
        data((m x n) 2-d numpy array): a data set of m instances and n features
        no_of_clusters(integer): the number of clusters
        centroids_initial(): the optional initial values for the centroids
    
    Returns:
        data((m x (n + 1)) 2-d numpy array): the data set with one more column that contains the vector's cluster
        centroids_new((k x n)2-d numpy array): contains the k = no_of_clusters centroids with n features
        centroids_history((l x 2) 2-d numpy array): an array to keep the previous positions of the centroids for 
                                                    better visualisation of the result. 

    '''
    
    # Initializations
    N = len(data)
    no_of_features = len(data[0])
    # if the centroid are provided as parameter use them, otherwise create them
    if centroids_initial == None:
        centroids_old = np.random.choice(np.arange(np.min(data), np.max(data), 0.1), size = (no_of_clusters, no_of_features), replace = False)
    else:
        centroids_old = centroids_initial
    centroids_new = np.zeros(centroids_old.shape) 
    centroids_history = np.copy(centroids_old) # in this array we stack the old positions of the centroids
    
    # A do - while loop implementation in Python, as the loop needs to run at least once
    condition = True
    
    while condition:
        distances_from_repr = np.zeros((N, 0)) # new every time, because we need to empty it
        # Determine the closest representative
        for i, centroid in enumerate(centroids_old):
            eucl_dist = euclidean_distance(data, centroid)
            distances_from_repr = np.concatenate((distances_from_repr, eucl_dist), axis = 1)
            
        nearest_cluster = np.argmin(distances_from_repr, axis = 1)
        
        # Parameter Updating
        for i, centroid in enumerate(centroids_old):
            indices_of_current_centroid = np.where(nearest_cluster == i)[0]
            if len(indices_of_current_centroid) == 0:
                centroids_new[i] = centroid # if no vector belongs to the centroid, leave the vector where it is
            else:
                centroids_new[i] = np.mean(data[indices_of_current_centroid, :], axis = 0)
        
        # Update the termination criterion where e = 0.00001
        criterion_array = np.absolute(centroids_new - centroids_old) < 0.00001
        if np.all(criterion_array) :
            condition = False
            data = np.hstack((data, nearest_cluster.reshape(N, 1)))#sort of: if we are going to exit the loop, take the clusters with you with the data matrix
        
        centroids_old = np.copy(centroids_new)
        centroids_history = np.vstack((centroids_history, centroids_old))
    
    return data, centroids_new, centroids_history
    
    
    
        
    