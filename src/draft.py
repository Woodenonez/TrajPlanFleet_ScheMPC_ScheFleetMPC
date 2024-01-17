from scipy.spatial import ConvexHull, convex_hull_plot_2d
import numpy as np
import matplotlib.pyplot as plt

# This is the quick hull algorithm

generators = np.array([[0.2, 0.2],
                       [0.2, 0.4],
                       [0.4, 0.4],
                       [0.4, 0.2],
                       [0.3, 0.5],
                       [0.3, 0.6]])
hull = ConvexHull(points=generators,
                  qhull_options='QG5')

print(hull.simplices)
print(hull.good)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for visible_facet in hull.simplices[hull.good]:
    ax.plot(hull.points[visible_facet, 0],
            hull.points[visible_facet, 1],
            color='violet',
            lw=6)
convex_hull_plot_2d(hull, ax=ax)
plt.show()