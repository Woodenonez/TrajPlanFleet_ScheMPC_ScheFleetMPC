import timeit

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d

# This is the quick hull algorithm

def get_visible_facets(point: np.ndarray, polygon: np.ndarray):
    """Get the 2D visible facets of a point in a polygon.

    Args:
        point: A target point.
        polygon: A n-degree polygon in the form of a nx2 arrays.

    Returns:
        visible_facet_coords: The coordinates of m visible facets, mxnx2.
        generator_hull: The convex hull of the generators (including the point and the polygon).
    """
    pt = np.array(point).reshape(1, -1)
    generators = np.concatenate((pt, polygon), axis=0)
    polygon_hull = ConvexHull(points=generators, qhull_options='QG0')

    visible_facets = polygon_hull.simplices[polygon_hull.good]
    visible_facet_coords = polygon_hull.points[visible_facets]
    return visible_facet_coords, polygon_hull

def get_edge_points_from_facets(point: np.ndarray, facet_coords: np.ndarray):
    """Get the 2D edge points of a facet that is visible to a point.

    Args:
        point: A target point.
        facet_coords: The coordinates of m visible facets, mxnx2.

    Returns:
        edge_points: The two edge points of the facet that is visible to the point.
    """
    pt = np.array(point).reshape(1, -1)
    unique_facet_points = np.unique(facet_coords.reshape(-1, 2), axis=0)
    vector = unique_facet_points - pt
    angles = np.arctan2(vector[:, 1], vector[:, 0])
    edge_points = unique_facet_points[[np.argmin(angles), np.argmax(angles)], :]
    return edge_points


point = np.array([0.3, 0.6])
polygon_1 = np.array([[0.2, 0.2],
                      [0.2, 0.4],
                      [0.4, 0.4],
                      [0.4, 0.2],
                      [0.3, 0.5]]) # 5
polygon_2 = np.array([[0.2, 0.2],
                      [0.1, 0.3],
                      [0.1, 0.4],
                      [0.5, 0.4],
                      [0.5, 0.3],
                      [0.4, 0.2],
                      [0.25, 0.5],
                      [0.35, 0.5]]) # 8

test_polygon = polygon_2

repeat = 10000
t1 = timeit.timeit(lambda: get_visible_facets(point, test_polygon), number=repeat)
t2 = timeit.timeit(lambda: get_edge_points_from_facets(point, test_polygon), number=repeat)
print(f"get_visible_facets: {t1/repeat*1000:.2f} ms")
print(f"get_edge_points_from_facets: {t2/repeat*1000:.2f} ms")

visible_facet_coords, hull = get_visible_facets(point, test_polygon)
edge_points = get_edge_points_from_facets(point, visible_facet_coords)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for coord in visible_facet_coords:
    ax.plot(coord[:, 0], coord[:, 1], color='violet', lw=6)
convex_hull_plot_2d(hull, ax=ax)
ax.plot(edge_points[:, 0], edge_points[:, 1], 'x', color='red')
ax.axis('equal')
plt.show()