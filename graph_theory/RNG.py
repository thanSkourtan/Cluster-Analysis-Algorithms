import numpy as np
import matplotlib.pyplot as plt


euclidean_distance = lambda data, point: np.sqrt(np.sum(np.power(data - point, 2), axis = 1).reshape((len(data), 1)))

def relative_neighborhood_graphs(data):
    
    #debug:
    #plt.gca().set_aspect('equal', adjustable='box')
    
    #construct the complete Graph G
    N = len(data)
    G = np.empty((N, 0)) 
    for point in data:
        G = np.concatenate((G, euclidean_distance(data,point)), axis=1)
        
        
    ###############
    RNG = np.zeros(G.shape)
    
    #O(n^3) time
    for i, point1 in enumerate(data):
        for j, point2 in enumerate(data):
            
            if i != j:
                flag = 1
                for k, pointK in enumerate(data):
                    if k != i and k !=j:
                        if max(G[i, k], G[j,k]) < G[i, j]:
                            flag = 0
                            break
                if flag:
                    #form an edge
                    RNG[i, j] = G[i, j]
    
                
    #debug
    #visualize RNG
    plt.scatter(data[:, 0], data[:, 1])
    x_, y_ = np.nonzero(RNG)
    for x, y in zip(x_, y_):
        if x < y:
            plt.plot((data[x, 0], data[y, 0]), (data[x, 1], data[y, 1]))
    plt.show()
    
    
    
    ##############################
    
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