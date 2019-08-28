# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
import pymintriangle


def compute_bounding_triangle(points, convex_hull=None):
    """
    Computes the minimum area enclosing triangle around a set of
    2D points.

    Parameters
    ----------
    points : (Mx2) array
        The coordinates of the points.
    convex_hull : scipy.spatial.ConvexHull, optional
        The convex hull of the points, as computed by SciPy.

    Returns
    -------
    triangle : polygon
        The minimum area enclosing triangle as a shapely polygon.
    """
    if convex_hull is None:
        convex_hull = ConvexHull(points)
    triangle = pymintriangle.compute(points[convex_hull.vertices])
    return Polygon(triangle)
