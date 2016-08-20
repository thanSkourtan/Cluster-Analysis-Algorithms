import numpy as np

#debug
import matplotlib.pyplot as plt


euclidean_distance = lambda data, point: np.sqrt(np.sum(np.power(data - point, 2), axis = 1).reshape((len(data), 1)))

def fuzzy(data, no_of_clusters, q = 1.25):
    #debug
    plt.scatter(data[:, 0], data[:, 1])
    partition_matrix = np.zeros((len(data), no_of_clusters))
    N = len(data)
    
    #initialize centroids 
    centroids_old = np.random.choice(np.arange(np.min(data), np.max(data)), size = (no_of_clusters, 2), replace = False)
    centroids_new = np.zeros(centroids_old.shape) 
    
    # A do - while loop implementation in Python, as the algorithm needs to run at least once
    condition = True
    
    while condition:
        # Update the U matrix
        for i in range(N):
            # Precalculate euclidean distances for the instance.
            eucl_dist = euclidean_distance(centroids_old, data[i,:])
            for j in range(no_of_clusters):
                partition_matrix[i][j] = 1 / np.sum(np.power((1./eucl_dist) * eucl_dist[j, 0], (1/(q-1))))
        
        # Update the centroids
        for i, centroid in enumerate(centroids_old):
            centroids_new[i] = np.sum(np.power(partition_matrix[:,[i]], q) * data,axis = 0) / np.sum(np.power(partition_matrix[:,i], q))
        
        # Update the termination criterion where e = 0.00001
        criterion_array = np.abs(centroids_new - centroids_old) < 0.00001
        if np.any(criterion_array) :
            condition = False
           
        centroids_old = np.copy(centroids_new)
        
        colors = ['g', 'r', 'm']
        for i, c in enumerate(centroids_new):
            plt.scatter(centroids_new[i, 0], centroids_new[i, 1], s = 80,c = colors[i], marker = 'x')

    plt.show()
    
    # Assign each vector to a cluster
    
    
    
    
    #ita calculation
    ita = []
    for centroid in centroids_new:
        ita.append(np.sum(np.power(partition_matrix[:, [0]],q) * euclidean_distance(data, centroid)) / np.sum(np.power(partition_matrix[:, 0], 2)))
    
    
    return partition_matrix, centroids_new, ita
    


