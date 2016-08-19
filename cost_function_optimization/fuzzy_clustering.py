import numpy as np

#debug
import matplotlib.pyplot as plt


def fuzzy(data, no_of_clusters, q = 1.25):
    
    #debug
    plt.scatter(data[:, 0], data[:, 1], c = 'b')
    partition_matrix = np.zeros((len(data), no_of_clusters))
    #initialize centroids to 0
    centroids = np.random.rand(no_of_clusters, 2)
    #centroids = np.array([[6., 1379.],[5., 817.]])
    #centroids = np.array([[9, 1]])
    #euclidean between data array and point
    euclidean_distance = lambda data, point: np.sqrt(np.sum(np.power(data - point, 2), axis = 1).reshape((len(data), 1)))
    
    #debug
    colors = ['g', 'r', 'm']
    #add the termination criterion
    t = 0
    while(t < 30):
        for i in range(len(data)):
            #precalculate euclidean distances for the instance. instance is the point, centroids are the data
            eucl_dist = euclidean_distance(centroids, data[i,:])
            for j in range(no_of_clusters):
                partition_matrix[i][j] = 1 / np.sum(np.power((1./eucl_dist) * eucl_dist[j, 0], (1/(q-1))))
        
        #update the centroids
        for i, centroid in enumerate(centroids):
            centroids[i] = np.sum(np.power(partition_matrix[:,[i]], q) * data,axis = 0) / np.sum(np.power(partition_matrix[:,i], q))
            
        t += 1
        #debug
        print(centroids)
        
        plt.scatter(centroids[:, 0], centroids[:, 1], s = 35,c = 'r')
    plt.show()
    
    
    
    
    #ita calculation
    ita = []
    for centroid in centroids:
        ita.append(np.sum(np.power(partition_matrix[:, [0]],q) * euclidean_distance(data, centroid)) / np.sum(np.power(partition_matrix[:, 0], 2)))
    
    return partition_matrix, centroids, ita
    


