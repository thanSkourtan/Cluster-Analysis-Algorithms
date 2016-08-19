import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

np.random.seed(seed = 10)
points = np.random.randint(100, size = (5, 2))
print(points)
tri = Delaunay(points)

for point in points:
    print(tri.find_simplex(point))

print(tri.simplices)
print(tri.neighbors)



plt.triplot(points[:,0], points[:,1], tri.simplices.copy())
plt.plot(points[:,0], points[:,1], 'o')
plt.show()
