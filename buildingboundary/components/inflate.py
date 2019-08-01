# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import numpy as np
from shapely.geometry import Polygon, Point
from shapely.ops import nearest_points

from ..utils import create_segments, distance


def point_on_line_segment(line_segment, point):
    a = line_segment[0]
    b = line_segment[1]
    p = point
    if not np.isclose(np.cross(b-a, p-a), 0):
        return False
    else:
        dist_ab = distance(a, b)
        dist_ap = distance(a, p)
        dist_bp = distance(b, p)
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
    polygon = Polygon(vertices)
    edges = create_segments(new_vertices)
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
        if len(nearest) == 1:
            i = nearest[0]
            # Move polygon edge out such that point is enclosed
            delta = p - point_on_polygon
            new_vertices[i] += delta
            new_vertices[(i+1) % len(new_vertices)] += delta
        else:
            # Ignore if closest point is polygon vertex
            outliers_indices = np.where(outliers_mask)[0]
            outlier_idx = outliers_indices[np.argmax(distances)]
            points = np.delete(points, outlier_idx, axis=0)

        # Update polygon
        polygon = Polygon(new_vertices)
        edges = create_segments(new_vertices)
        outliers_mask = [not polygon.contains(Point(p)) for p in points]
        outliers = points[outliers_mask]

    return new_vertices
