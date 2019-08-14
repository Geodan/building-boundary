# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
import pymintriangle


def compute_bounding_triangle(points, convex_hull=None):
    if convex_hull is None:
        convex_hull = ConvexHull(points)
    triangle = pymintriangle.compute(points[convex_hull.vertices])
    return Polygon(triangle)
