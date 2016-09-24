import numpy as np
from scipy import ndimage
import os, inspect
from tqdm import tqdm
from functools import reduce
from sys import maxsize as max_integer
import matplotlib.pyplot as plt
from utility.plotting_functions import *

'''
Module of utility functions set to work with the image repository of 
The Berkeley Segmentation Dataset and Benchmark.
'''



def insert_clusters(original_image, seg_file):
    ''' A function that takes the seg format files along with the original image 
        and returns the image as a numpy array, ALONG with the externaly provided 
        clusters (called segments in the seg file).
    '''
    # Find the project's directory and get the files
    this_file = os.path.abspath(inspect.getfile(inspect.currentframe()))
    project_dir = os.path.dirname(os.path.dirname(this_file))
    path_to_images = os.path.join(project_dir,'images\\')
    
    labels = np.loadtxt(path_to_images + seg_file, skiprows = 11)
    
    image = ndimage.imread(path_to_images + original_image)
    
    # Add one extra column to image array in order to keep the cluster ids
    temp = np.zeros((image.shape[0], image.shape[1], 1))
    clustered_image = np.dstack((image, temp))
    
    for row in labels:
        clustered_image[row[1], row[2]: row[3], 3] = row[0]

    return clustered_image
    
    
    
def rand_index_calculation(X_, external_info):
    initial_shape = list(X_.shape)
    N = reduce(lambda x, y: x * y, X_.shape[:-1])
    m = X_.shape[-1] 
    X_ = X_.reshape(N, m)
        
    external_info = external_info.reshape(N, m)
        
    random_size = 5000
    indices = np.random.choice(len(X_), size = random_size, replace = False)
    X_random = X_[indices,:]
    external_info_random = external_info[indices, :]
        
    nominator = 0
    for i, vector1 in tqdm(enumerate(X_random)):
        temp = external_info_random[i + 1:, :]
        for j, vector2 in enumerate(temp, start = i + 1):
            if ((vector1[3] == X_random[j, 3] and vector2[3] == external_info_random[i, 3]) or 
                (vector1[3] != X_random[j, 3] and vector2[3] != external_info_random[i, 3])):
                nominator += 1
  
    X_ = X_.reshape(initial_shape)
    rand_index = nominator/(random_size*(random_size-1)/2)
    return rand_index
        






def merging_procedure(image, threshold):
    

    N = image.shape[0]
    m = image.shape[1]
    visited = np.zeros((N, m))
    
    no_of_clusters = len(np.unique(image[:,:,3]))
    

    for index in tqdm(np.ndindex(N, m)):
        
        if(visited[index[0], index[1]] == 0):
            dominant_cluster_list = np.zeros(no_of_clusters + 1) # Plus one in case some algorithm starts numbering the clusters fron 1 instead of zero
            counter = 1
            visited[index[0], index[1]] = 3 
            counter, dominant_cluster_list = _dfs_util(image, index[0], index[1], N, m, visited, image[index[0], index[1], 3], counter, dominant_cluster_list, threshold)
            # Reset visited array
            indices_of_dominant_clusters = np.where(visited==2)
            visited[indices_of_dominant_clusters[0], indices_of_dominant_clusters[1]] = 0
            
            indices_of_previously_visited = np.where(visited == 3)
            visited[indices_of_previously_visited[0], indices_of_previously_visited[1]] = 1
            
            if counter < threshold:
                dom_cluster = np.argmax(dominant_cluster_list)
                
                # Change all pixels of previous island to the dominant cluster
                image[indices_of_previously_visited[0], indices_of_previously_visited[1], 3] = dom_cluster
                #draw_clustered_image(image, (321, 481, 3), 5)
             
        
        
    return image


def _moves(y, x):
    moves = [(0,1), (1,0), (1,1), (-1, -1), (-1, 0),(-1, 1),(0,-1),(1,-1) ]
    list_of_new_positions = []
    for move in moves:
        list_of_new_positions.append((y + move[0], x + move[1]))    
    return list_of_new_positions

def _constraints(y, x, N, m): 
    if y >= N or x >= m:
        return False 
    elif y < 0 or x < 0:
        return False
    else:
        return True

def _dfs_util(image, y, x, N, m, visited, pixels_cluster, counter, dominant_cluster_list, threshold):
    
    if counter > threshold:
        return max_integer, []
    
    for move in _moves(y, x):
        y = move[0]
        x = move[1]
        if _constraints(y,x, N, m) == True:
            #print('y', y)
            #print('x', x)
            if visited[y, x] == 0 or visited[y, x] == 2:
                if image[y, x, 3] == pixels_cluster:
                    visited[y, x] = 3
                    counter += 1
                    counter, dominant_cluster_list = _dfs_util(image, y, x, N, m, visited, image[y, x, 3], counter, dominant_cluster_list, threshold)
                    if counter == max_integer:
                        return counter, dominant_cluster_list
                else:
                    if visited[y, x] != 2:
                        #print('boundary y ', y)
                        #print('boundary x ', x)
                        visited[y, x] = 2 # 2 means not visited by used for . it's a workaround in order not to use another array
                        dominant_cluster_list[image[y, x, 3]] += 1
            else: # when we reach here, the pixel is external. External means boundary to other cluster not the image limits
                if visited[y, x] != 2 and visited[y, x] != 3:
                    #print('boundary y ', y)
                    #print('boundary x ', x)
                    visited[y, x] = 2 # 2 means not visited by used for . it's a workaround in order not to use another array
                    dominant_cluster_list[image[y, x, 3]] += 1
    return counter, dominant_cluster_list
                
    























