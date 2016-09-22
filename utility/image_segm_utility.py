import numpy as np
from scipy import ndimage
import os, inspect
from tqdm import tqdm
from functools import reduce

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
        
        