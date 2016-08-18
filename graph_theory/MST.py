import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import *
from sys import maxsize as max_integer

euclidean_distance = lambda data, point: np.sqrt(np.sum(np.power(data - point, 2), axis = 1).reshape((len(data), 1)))


def minimum_spanning_tree(data, k = 5, q = 2, f = 2):
    
    #debug:
    #plt.gca().set_aspect('equal', adjustable='box')
    
    #construct the complete Graph G
    N = len(data)
    G = np.empty((N, 0)) 
    for point in data:
        G = np.concatenate((G, euclidean_distance(data,point)), axis=1)
    
    ######################
    
    MST = np.zeros(G.shape)
    
    N = len(G)
    
    key = np.zeros(N)
    key.fill(max_integer)
    parent = np.zeros(N)
    
    key[0] = 0
    parent[0] = -1
    
    visited = np.zeros(N)
    
    for i in range(N - 1): #edges
        
        current_value = np.min(key[np.where(visited == 0)]) #take the min key where visited  = 0
        
        current_node = -1
        for i, element in enumerate(key):
            if element == current_value and visited[i] == 0:
                current_node = i 
                break
        
        visited[current_node] = 1
        
        for adj_node in range(N): #all nodes are adjacent
            
            if  G[current_node, adj_node] < key[adj_node] and visited[adj_node] == 0 and G[current_node][adj_node] !=0:
                key[adj_node] = G[current_node, adj_node]
                parent[adj_node] = current_node
                
    
    #fill in MST table
    x_list = []
    y_list = []
    for i, j in enumerate(zip(key[1:], parent[1:]), start = 1):
        MST[j[1], i] = j[0]
        x_list.append(j[1])
        y_list.append(i)
        MST[i, j[1]] = j[0]
        
      
    #debug: visualize MST
    
    
    #plt.scatter(data[:, 0], data[:, 1])
    x, y= np.nonzero(MST)
    
    for ole in zip(x, y):    
        plt.plot((data[ole[0], 0], data[ole[1], 0]), (data[ole[0], 1], data[ole[1], 1]), color = 'm')    
    #plt.show()
    
    
    ##########################
    
    
   
    inconsistent = np.zeros(N - 1) #follows x_list, y_list
    
    #find all pairs of nodes of edges
    for i, nodes in enumerate(zip(x_list, y_list)):
        weight = MST[nodes[0], nodes[1]]
        list_of_weights_N1 = np.empty((0, 0))
        list_of_weights_N2 = np.empty((0, 0))
        
        list_of_weights_N1 = _recursion_util(nodes, k, list_of_weights_N1, MST)
        list_of_weights_N2 = _recursion_util(nodes[::-1], k, list_of_weights_N2, MST)
        
        #inconsistency criterion
        weight_mean_N1 = np.mean(list_of_weights_N1)
        weight_mean_N2 = np.mean(list_of_weights_N2)
        weight_std_N1 = np.std(list_of_weights_N1)
        weight_std_N2 = np.std(list_of_weights_N2)
        
        if weight > max(q * weight_std_N1 + weight_mean_N1, q * weight_std_N2 + weight_mean_N2) and \
           weight / max(weight_mean_N1, weight_mean_N2) > f: 
            inconsistent[i] = 1 
    
    
    #debug: show in graph inconsistent edges
    for i, inc in enumerate(inconsistent):
        if inc == 1:
            plt.text((data[x_list[i],0] + data[y_list[i],0])/2, (data[x_list[i],1]  + data[y_list[i],1])/2 , "inc")
    
    #debug: show in graph weights
    '''for ind, g in enumerate(MST):
        for oe, weight in enumerate(g):
            if weight !=0:
                plt.text((data[ind,0] + data[oe, 0])/2, (data[ind, 1]  + data[oe, 1])/2 , round(weight,2), fontsize=7)
    '''
    
    ##########################

    data = np.hstack((data, np.zeros((len(data), 1))))
    
    #get the indices where inconsistent is not zero
    inc_edges_indices = np.nonzero(inconsistent)
    
    for index in inc_edges_indices[0]:
        MST[x_list[index], y_list[index]] =MST[y_list[index], x_list[index]] = 0
    
    visited_nodes = np.zeros(N)
    
    #dfs from now on
    cluster_id = 1
    for s in range(N):
        if(visited_nodes[s] == 0):
            visited_nodes[s] = 1
            data[s, 2] = cluster_id
            _dfs_util(MST, s, visited_nodes, cluster_id, data)
        cluster_id += 1
    
    return data

    
def _dfs_util(MST, s, visited_nodes, cluster_id, data):
    adj_nodes =  np.nonzero(MST[s, :])
    for node in adj_nodes[0]:
        if visited_nodes[node] == 0:
            visited_nodes[node] = 1
            data[node, 2] = cluster_id
            _dfs_util(MST, node, visited_nodes, cluster_id, data)
    
        
def _recursion_util(nodes, k, list_of_weights, MST):
    
    if k == 0: return list_of_weights
    
    current_node1 = nodes[0]
    current_node2 = nodes[1]
    
    adj_nodes = np.nonzero(MST[current_node1, :])
    adj_nodes = np.delete(adj_nodes, np.where(adj_nodes[0] == current_node2))
    
    if  len(adj_nodes) == 0: return list_of_weights
    
    for node in adj_nodes:
        k = k - 1
        list_of_weights = np.append(list_of_weights, MST[node, current_node1])
        list_of_weights = _recursion_util((node, current_node1),k, list_of_weights, MST)
        k += 1
    
    return list_of_weights 
    

