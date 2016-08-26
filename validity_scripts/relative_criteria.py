import numpy as np
from tqdm import tqdm
from cost_function_optimization import fuzzy_clustering
from sys import maxsize as max_integer
import matplotlib.pyplot as plt


def relative_validity(X):
    
    # Initialization
    no_of_clusters_list = [i for i in range(2, 11)]
    values_of_q = [1.3, 1.5, 2, 2.5, 3, 3.5, 5]
    
    # Initialize arrays to hold the indices. We use separate arrays for easier modification of the code if needed.
    # If we wanted to use one array then this would be a 3 - dimensional array.
    PC = np.zeros((len(no_of_clusters_list), len(values_of_q)))
    PE = np.zeros((len(no_of_clusters_list), len(values_of_q)))
    XB = np.zeros((len(no_of_clusters_list), len(values_of_q)))
    FS = np.zeros((len(no_of_clusters_list), len(values_of_q)))
    
    for i, total_clusters in tqdm(enumerate(no_of_clusters_list)): # no_of_clusters
        for j, q_value in enumerate(values_of_q): #edw vazw to q
            # When X returns it has one more column that needs to be erased
            X_, centroids, ita, centroids_history, partition_matrix = fuzzy_clustering.fuzzy(X, total_clusters, q = q_value)
            
            
            
            # Calculate index
            PC[i, j] = partition_coefficient(X, partition_matrix)
            #print(PC[i,j])
            #print(np.sum(np.power(partition_matrix, 2), axis = 1))
            #plot_data_util(X_, centroids, centroids_history ,total_clusters)
            #plt.show()
            PE[i, j] = partition_entropy(X, partition_matrix)
            XB[i, j] = Xie_Beni(X, centroids, partition_matrix)
            FS[i, j] = fukuyama_sugeno(X, centroids, partition_matrix, q = 2)
        
    return no_of_clusters_list, values_of_q, PC, PE, XB, FS
        
    



partition_coefficient = lambda X, partition_matrix: np.round(1/len(X) * np.sum(np.power(partition_matrix, 2)), 5)
partition_entropy = lambda X, partition_matrix: - 1/len(X) * np.sum(partition_matrix * np.log(partition_matrix)) 


def Xie_Beni(X, centroids, partition_matrix):
    
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
            if k != l:
                temp = centroid1 - centroid2
                distance = np.sum(np.power(temp, 2)) # it will always be 1 x 1, euclidean distance without the root
                if min_distance > distance:
                    min_distance = distance 
                
    Xie_Beni = total_variation/(min_distance * len(X))
    return Xie_Beni


def fukuyama_sugeno(X, centroids, partition_matrix, q = 2):
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



def plot_data_util(X, centroids, centroids_history ,no_of_clusters):
    
    # Initialization
    np.random.seed(seed = None)
    clusters = np.unique(X[:, 2])
    
    # Initialize plots
    f, initDataPlot = plt.subplots(2, sharex=True,  figsize = (12,8))
    f.canvas.set_window_title('Unclustered and Clustered Data')
    plt.tight_layout()

    initDataPlot[0].set_title('Initial Data')
    initDataPlot[0].set_xlabel('Feature 1')
    initDataPlot[0].set_ylabel('Feature 2')
    
    initDataPlot[1].set_title('Clustered Data')
    initDataPlot[1].set_xlabel('Feature 1')
    initDataPlot[1].set_ylabel('Feature 2')
    
    
    # Plot initial data set without clustering
    initDataPlot[0].scatter(X[:, 0], X[:, 1])
    
    # Plot data after clustering
    for i, cluster in enumerate(clusters):
        initDataPlot[1].scatter(X[ X[:,2] == cluster, 0], X[ X[:, 2] == cluster, 1], c=np.random.rand(3,1), s = 30)
    
    # Plots the centroids history
    colors= ['k', 'b', 'g', 'y', 'm', 'c']
    for alpha_counter, i in enumerate(range(0, len(centroids_history),  no_of_clusters)):
        for j in range(i, i + no_of_clusters):
            initDataPlot[1].plot(centroids_history[j, 0], centroids_history[j, 1], c = colors[j % len(colors)], marker = 'x', mew =  1, ms = 15, alpha = 0.2 + alpha_counter * 0.8/(len(centroids_history)/no_of_clusters))
            
    # Plots the centroids
    for i, c in enumerate(centroids):
            initDataPlot[1].plot(centroids[i, 0], centroids[i, 1], c = 'r', marker = 'x', mew=2, ms = 10)
 


















