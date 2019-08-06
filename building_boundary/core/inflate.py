# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import numpy as np
from shapely.geometry import Polygon, Point
from shapely.ops import nearest_points

from .segment import BoundarySegment
from .. import utils


def point_on_line_segment(line_segment, point):
    a = line_segment[0]
    b = line_segment[1]
    p = point
    if not np.isclose(np.cross(b-a, p-a), 0):
        return False
    else:
        dist_ab = utils.geometry.distance(a, b)
        dist_ap = utils.geometry.distance(a, p)
        dist_bp = utils.geometry.distance(b, p)
        if np.isclose(dist_ap + dist_bp, dist_ab):
            return True
        else:
            return False


def nearest_edges(point_on_polygon, edges):
    nearest = []
    for i, e in enumerate(edges):
        if point_on_line_segment(e, point_on_polygon):
            nearest.append(i)
    return nearest


def inflate_polygon(vertices, points):
    # Find points not enclosed in polygon
    new_vertices = vertices.copy()
    n_vertices = len(vertices)
    polygon = Polygon(vertices)
    edges = utils.create_pairs(new_vertices)
    outliers_mask = [not polygon.contains(Point(p)) for p in points]
    outliers = points[outliers_mask]

    while len(outliers) > 0:
        # Get furthest point
        distances = [polygon.distance(Point(p)) for p in outliers]
        if np.isclose(max(distances), 0):
            break
        p = outliers[np.argmax(distances)]
        # Find nearest polygon edge to point
        point_on_polygon, _ = nearest_points(polygon, Point(p))
        point_on_polygon = np.array(point_on_polygon)
        nearest = nearest_edges(point_on_polygon, edges)
        for i in nearest:
            # Move polygon edge out such that point is enclosed
            delta = p - point_on_polygon
            p1 = new_vertices[i] + delta
            p2 = new_vertices[(i+1) % n_vertices] + delta
            l1 = BoundarySegment(np.array([new_vertices[(i-1) % n_vertices],
                                           new_vertices[i]]))
            l2 = BoundarySegment(np.array([p1, p2]))
            l3 = BoundarySegment(np.array([new_vertices[(i+1) % n_vertices],
                                           new_vertices[(i+2) % n_vertices]]))
            new_vertices[i] = l2.line_intersect(l1.line)
            new_vertices[(i+1) % n_vertices] = l2.line_intersect(l3.line)

        # Update polygon
        polygon = Polygon(new_vertices)
        edges = utils.create_pairs(new_vertices)
        outliers_mask = [not polygon.contains(Point(p)) for p in points]
        outliers = points[outliers_mask]

    return new_vertices
