import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


euclidean_distance = lambda data, point: np.sqrt(np.sum(np.power(data - point, 2), axis = 1).reshape((len(data), 1)))

np.random.seed(seed=400)
n = 60

data = np.random.randint(n, size = (n, 2))

# We are going to plot the cost against all possible values of theta
theta1_x = np.arange(0, n)
theta1_y = np.arange(0, n)

# Build the weight matrix
q = 3
U = np.random.rand(n, 1)
temp = np.ones((n, 1))
U = np.power(np.hstack((U, temp-U)), q)

cost = []
centroid1_dimensions = np.zeros((np.power(len(theta1_x), 4), 2))
centroid2_dimensions = np.zeros((np.power(len(theta1_x), 4), 2))
i = 0 
j = 0
for x in theta1_x:
    for y in theta1_y:
        for k in theta1_x:
            for m in theta1_y:
                centroid1 = np.array([[x, y]])
                centroid2 = np.array([[k, m]])
                
                centroid1_dimensions[i, 0] = x
                centroid1_dimensions[i, 1] = y
                centroid2_dimensions[j, 0] = k
                centroid2_dimensions[j, 1] = m
                i = i + 1
                j = j + 1
                
                dist1 = euclidean_distance(data, centroid1)
                dist2 = euclidean_distance(data, centroid2)
                
                distances = np.hstack((dist1, dist2)).T
                cost.append(np.sum(np.dot(U, distances)))


lala = np.array(centroid1_dimensions)
np.array(centroid2_dimensions)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(lala[:,0], lala[:,1], cost, cmap=cm.jet, linewidth=0.2)
plt.show()
