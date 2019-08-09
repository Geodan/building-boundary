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
    new_vertices = vertices.copy()
    points_geom = [Point(p) for p in points]
    n_vertices = len(vertices)
    polygon = Polygon(vertices)
    edges = utils.create_pairs(new_vertices)

    # Find points not enclosed in polygon
    distances = np.array([polygon.distance(p) for p in points_geom])
    outliers_mask = np.invert(np.isclose(distances, 0))
    outliers = points[outliers_mask]
    distances = distances[outliers_mask]
    n_outliers = len(outliers)

    while n_outliers > 0:
        p = outliers[np.argmax(distances)]

        # Find nearest polygon edge to point
        point_on_polygon, _ = nearest_points(polygon, Point(p))
        point_on_polygon = np.array(point_on_polygon)
        nearest = nearest_edges(point_on_polygon, edges)

        # Move polygon edge out such that point is enclosed
        for i in nearest:
            delta = p - point_on_polygon
            p1 = new_vertices[i] + delta
            p2 = new_vertices[(i+1) % n_vertices] + delta
            # Lines
            l1 = BoundarySegment(np.array([new_vertices[(i-1) % n_vertices],
                                           new_vertices[i]]))
            l2 = BoundarySegment(np.array([p1, p2]))
            l3 = BoundarySegment(np.array([new_vertices[(i+1) % n_vertices],
                                           new_vertices[(i+2) % n_vertices]]))
            # Intersections
            i1 = l2.line_intersect(l1.line)
            i2 = l2.line_intersect(l3.line)

            new_vertices[i] = i1
            new_vertices[(i+1) % n_vertices] = i2

            # Update polygon
            polygon = Polygon(new_vertices)
            edges = utils.create_pairs(new_vertices)
            point_on_polygon, _ = nearest_points(polygon, Point(p))
            point_on_polygon = np.array(point_on_polygon)

        distances = np.array([polygon.distance(p) for p in points_geom])
        outliers_mask = np.invert(np.isclose(distances, 0))
        outliers = points[outliers_mask]
        distances = distances[outliers_mask]

        if len(outliers) >= n_outliers:
            break
        n_outliers = len(outliers)

    if not Polygon(new_vertices).is_valid and Polygon(vertices).is_valid:
        return vertices
    else:
        return new_vertices
