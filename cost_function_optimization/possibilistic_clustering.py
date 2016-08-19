import numpy as np
import matplotlib.pyplot as plt

def possibilistic_clustering(data, no_of_clusters, q, ita):
    
    parameter_matrix = np.zeros((len(data), no_of_clusters))
    #initialize centroids to 0
    #centroids = np.zeros((no_of_clusters, 2))
    centroids = np.array([[6., 1379.],[5., 817.]])
    
    
    
    #euclidean between data array and point
    euclidean_distance = lambda data, point: np.sqrt(np.sum(np.power(data - point, 2), axis = 1).reshape((len(data), 1)))
    
    #add the termination criterion
    t = 0
    while(t < 50):
        for i in range(len(data)):
            #precalculate euclidean distances for the instance. instance is the point, centroids are the data
            eucl_dist = euclidean_distance(centroids, data[i,:])
            for j in range(no_of_clusters):
                parameter_matrix[i][j] = 1 / (1 + np.power(eucl_dist[j][0]/ita[j], (1/(q-1))))
        
        #update the centroids
        for i, centroid in enumerate(centroids):
            centroids[i] = np.sum(np.power(parameter_matrix[:,[i]], q) * data,axis = 0) / np.sum(np.power(parameter_matrix[:,i], q))
            print(centroids[i])
        t += 1
        #debug
        plt.scatter(data[:, 0], data[:, 1])
        plt.scatter(centroids[:, 0], centroids[:, 1], marker = 'x', c = 'r')
    
    plt.show()
    