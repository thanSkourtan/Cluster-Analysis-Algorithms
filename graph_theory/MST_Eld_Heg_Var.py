import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import *
from sys import maxsize as max_integer
from scipy.spatial import Delaunay

euclidean_distance = lambda data, point: np.sqrt(np.sum(np.power(data - point, 2), axis = 1).reshape((len(data), 1)))


def minimum_spanning_tree_variation(data):
    print(data)
    #debug:
    #plt.gca().set_aspect('equal', adjustable='box')
    
    
    
    N = len(data)
    G = np.zeros((N, N), np.float32)
    p = max_integer
    inconsistent = np.zeros(N - 1) #follows x_list, y_list
    
    tri = Delaunay(data)
    
    #debug: visualise delaunay triangulisation
    
    plt.triplot(data[:,0], data[:,1], tri.simplices.copy())
    plt.plot(data[:,0], data[:,1], 'o')
    for i, d in enumerate(data):
        plt.text(d[0], d[1], i)
    plt.show()
    
    
    for i, point in enumerate(data):
        all_triangles = np.where(tri.simplices[:,] == i) #all triangles that contain the point
        all_points_in_triangles = tri.simplices[all_triangles[0]]
        neighborhood_points = np.unique(all_points_in_triangles)
        #debug
        #print(neighborhood_points)
        neighborhood_distance_matrix = euclidean_distance(data[neighborhood_points, :], point)
        #debug
        #print(neighborhood_distance_matrix)
        for k, j in enumerate(neighborhood_points):
            if i !=j and neighborhood_distance_matrix[k] < p:
                G[i, j] = neighborhood_distance_matrix[k]
            
    #debug
    #print(G)  
    ##########################
    
    print('lala')
    #build list of edges. the edges are duplicates since G[i,j] = G[j,i], so keep only
    #those for which x_list element < y_list element 
    x_list, y_list = np.nonzero(G)
    all_edges = np.vstack((x_list, y_list))
    all_edges = all_edges[:, np.where(all_edges[0, :] < all_edges[1, :])[0]]
    
    
    #in order to find p we first need to define the delaunay triangulation graph
    edges_weights = G[all_edges[0, :], all_edges[1, :]]
    
    edges_weights = np.sort(edges_weights)
    
    #make 20 partitions
    step = int(np.ceil(len(edges_weights)/20))
    min_total = max_integer
    p_minimizing_total = 0
    
    max_diff = -8
    for i in range(0, len(edges_weights), step):
        first_partition = edges_weights[:i+step]
        second_partition = edges_weights[i+step:] 
        #tou siggrafea
        '''first_sum = np.sum(np.power((np.log(first_partition) - np.mean(np.log(first_partition))), 2))/(len(first_partition) - 1)
        second_sum = np.sum(np.power((np.log(second_partition) - np.mean(np.log(second_partition))), 2))/(len(second_partition) - 1)
        print(first_sum + second_sum)
        if min_total > (first_sum + second_sum):
            min_total = first_sum + second_sum
            print(min_total)
            p_minimizing_total = first_partition[-1]
        '''
        #diko mou
        ####################
        if max_diff < np.mean(second_partition) - np.mean(first_partition):
            max_diff = np.mean(second_partition) - np.mean(first_partition)
            p_minimizing_total = first_partition[-1]
        
        #len(second_partition)
    
    print('lala')
        
    #############################
    
    #exw to graph kai exw kai to p. auto pou ekremmei 
    #einai na kanw 0 sto G ta inconsistent edges kai 
    #meta me to recursion pou idi exw na vrw ta clusters
    #erase long edges
    #p_minimizing_total = 0.05
    
    
    
    print('p_minimizing: ', p_minimizing_total)
    G[np.where(G> p_minimizing_total)] = 0
    
    
    
    data = np.hstack((data, np.zeros((len(data), 1))))
    
    visited_nodes = np.zeros(N)
    
    #dfs from now on
    cluster_id = 1
    for s in range(N):
        if(visited_nodes[s] == 0):
            visited_nodes[s] = 1
            data[s, 2] = cluster_id
            _dfs_util(G, s, visited_nodes, cluster_id, data)
            cluster_id += 1
            
    
    return data

    
def _dfs_util(MST, s, visited_nodes, cluster_id, data):
    adj_nodes =  np.nonzero(MST[s, :])
    for node in adj_nodes[0]:
        if visited_nodes[node] == 0:
            visited_nodes[node] = 1
            data[node, -1] = cluster_id
            _dfs_util(MST, node, visited_nodes, cluster_id, data)
    
        

