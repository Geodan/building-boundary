# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import math

import numpy as np
from shapely.geometry import Polygon, MultiPolygon

try:
    from CGAL.CGAL_Kernel import Point_2
    from CGAL.CGAL_Alpha_shape_2 import Alpha_shape_2
    from CGAL.CGAL_Alpha_shape_2 import REGULAR
    CGAL_AVAILABLE = True
    unary_union = None
    Delaunay = None
except ImportError:
    from shapely.ops import unary_union
    from scipy.spatial import Delaunay
    CGAL_AVAILABLE = False
    Point_2 = None
    Alpha_shape_2 = None
    REGULAR = None


def alpha_shape_cgal(points, alpha):
    """
    Uses CGAL to compute the alpha shape (a concave hull) of a set of points.
    The alpha shape will not contain any interiors.

    Parameters
    ----------
    points : (Mx2) array
        The x and y coordinates of the points
    alpha : float
        Influences the shape of the alpha shape. Higher values lead to more
        triangles being deleted.

    Returns
    -------
    alpha_shape : polygon
        The computed alpha shape as a shapely polygon
    """
    points_cgal = [Point_2(*p) for p in points]

    as2 = Alpha_shape_2(points_cgal, 0, REGULAR)
    as2.set_alpha(alpha)

    edges = []
    for e in as2.alpha_shape_edges():
        segment = as2.segment(e)
        edges.append([[segment.vertex(0).x(), segment.vertex(0).y()],
                      [segment.vertex(1).x(), segment.vertex(1).y()]])
    edges = np.array(edges)

    e1s = edges[:, 0].tolist()
    e2s = edges[:, 1].tolist()
    polygons = []

    while len(e1s) > 0:
        polygon = []
        current_point = e2s[0]
        polygon.append(current_point)
        del e1s[0]
        del e2s[0]

        while True:
            try:
                i = e1s.index(current_point)
            except ValueError:
                break

            current_point = e2s[i]
            polygon.append(current_point)
            del e1s[i]
            del e2s[i]

        polygons.append(polygon)

    polygons = [Polygon(p) for p in polygons if len(p) > 2]

    alpha_shape = MultiPolygon(polygons).buffer(0)

    return alpha_shape


def triangle_geometry(triangle):
    """
    Compute the area and circumradius of a triangle.

    Parameters
    ----------
    triangle : (3x3) array-like
        The coordinates of the points which form the triangle.

    Returns
    -------
    area : float
        The area of the triangle
    circum_r : float
        The circumradius of the triangle
    """
    pa, pb, pc = triangle
    # Lengths of sides of triangle
    a = math.hypot((pa[0] - pb[0]), (pa[1] - pb[1]))
    b = math.hypot((pb[0] - pc[0]), (pb[1] - pc[1]))
    c = math.hypot((pc[0] - pa[0]), (pc[1] - pa[1]))
    # Semiperimeter of triangle
    s = (a + b + c) / 2.0
    # Area of triangle by Heron's formula
    area = math.sqrt(s * (s - a) * (s - b) * (s - c))
    if area != 0:
        circum_r = (a * b * c) / (4.0 * area)
    else:
        circum_r = 0
    return area, circum_r


def alpha_shape_python(points, alpha):
    """
    Compute the alpha shape (or concave hull) of points.
    The alpha shape will not contain any interiors.

    Parameters
    ----------
    points : (Mx2) array
        The coordinates of the points.
    alpha : float
        Influences the shape of the alpha shape. Higher values lead to more
        triangles being deleted.

    Returns
    -------
    alpha_shape : polygon
        The computed alpha shape as a shapely polygon
    """
    triangles = []
    tri = Delaunay(points)
    for t in tri.simplices:
        area, circum_r = triangle_geometry(points[t])
        if area != 0:
            if circum_r < 1.0 / alpha:
                triangles.append(Polygon(points[t]))

    alpha_shape = unary_union(triangles)
    if type(alpha_shape) == MultiPolygon:
        alpha_shape = MultiPolygon([Polygon(s.exterior) for s in alpha_shape])
    else:
        alpha_shape = Polygon(alpha_shape.exterior)

    return alpha_shape


def compute_alpha_shape(points, alpha):
    """
    Compute the alpha shape (or concave hull) of points.
    The alpha shape will not contain any interiors.

    Parameters
    ----------
    points : (Mx2) array
        The coordinates of the points.
    alpha : float
        Influences the shape of the alpha shape. Higher values lead to more
        triangles being deleted.

    Returns
    -------
    alpha_shape : polygon
        The computed alpha shape as a shapely polygon
    """
    if len(points) < 4:
        raise ValueError('Not enough points to compute an alpha shape.')
    if CGAL_AVAILABLE:
        alpha_shape = alpha_shape_cgal(points, alpha)
    else:
        alpha_shape = alpha_shape_python(points, alpha)
    return alpha_shape
