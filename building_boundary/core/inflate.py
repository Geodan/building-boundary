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
    """
    Determines if a point is on a line defined by two points.

    Parameters
    ----------
    line_segment : (2x2) array
        A line defined by the coordinates of two points.
    point : (1x2) array
        The coordinates of the point to check

    Returns
    -------
     : bool
        If the point is on the line defined by two points.
    """
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


def point_polygon_edges(point_on_polygon, edges):
    """
    Determines on which edge(s) of a polygon a point (which is on the polygon's
    exterior) lies.

    Parameters
    ----------
    point_on_polygon : (1x2) array
        The coordinates of the point.
    edges : list of tuple of (1x2) array
        Each tuple contains the coordinates of the start and end point of an
        edge of the polygon.

    Returns
    -------
    found_edges : list of int
        The indices of the edges on which the given point lies. Often of
        length 1, however if a point lies on a corner it lies on two edges
        and two indices will be returned.
    """
    found_edges = []
    for i, e in enumerate(edges):
        if point_on_line_segment(e, point_on_polygon):
            found_edges.append(i)
    return found_edges


def inflate_polygon(vertices, points):
    """
    Inflates the polygon such that it will contain all the given points.

    Parameters
    ----------
    vertices : (Mx2) array
        The coordinates of the vertices of the polygon.
    points : (Mx2) array
        The coordinates of the points that the polygon should contain.

    Returns
    -------
    vertices : (Mx2) array
        The coordinates of the vertices of the inflated polygon.
    """
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
        nearest_edges = point_polygon_edges(point_on_polygon, edges)

        # Move polygon edge out such that point is enclosed
        for i in nearest_edges:
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
            new_vertices[i] = l2.line_intersect(l1.line)
            new_vertices[(i+1) % n_vertices] = l2.line_intersect(l3.line)

            # Update polygon
            polygon = Polygon(new_vertices)

            if not polygon.is_valid:
                polygon = polygon.buffer(0)
                new_vertices = np.array(polygon.exterior.coords)
                n_vertices = len(new_vertices)
                n_outliers = float('inf')

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
