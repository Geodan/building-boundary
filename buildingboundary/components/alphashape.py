# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import numpy as np
from CGAL.CGAL_Kernel import Point_2
from CGAL.CGAL_Alpha_shape_2 import Alpha_shape_2
from CGAL.CGAL_Alpha_shape_2 import REGULAR
from shapely.geometry import Polygon, MultiPolygon


def compute_alpha_shape(points, alpha):
    """
    Uses CGAL to compute the alpha shape (a concave hull) of a set of points.
    The alpha shape will not contain any interiors.

    Parameters
    ----------
    points : (Mx2) array
        The x and y coordinates of the points
    alpha : float
        Influences the shape of the alpha shape. Higher values lead to more
        edges being deleted.

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
#    largest_polygon = max(polygons, key=lambda p: p.area)
#    alpha_shapes = [{'exterior': largest_polygon, 'interiors': []}]
#
#    for p in polygons:
#        for a in alpha_shapes:
#            if a['exterior'] != p and a['exterior'].contains(p):
#                a['interiors'].append(p)
#            elif a['exterior'] != p:
#                alpha_shape.append({'exterior': p, 'interiors': []})

    alpha_shape = MultiPolygon(polygons).buffer(0)

    return alpha_shape
