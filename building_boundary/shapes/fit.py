# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Point, Polygon
from shapely.ops import cascaded_union

import concave_hull

from .alpha_shape import compute_alpha_shape
from .bounding_box import compute_bounding_box
from .bounding_triangle import compute_bounding_triangle


def compute_shape(points, alpha=None, k=None):
    if alpha is not None:
        shape = compute_alpha_shape(points, alpha)

        if k is not None:
            boundary_points = concave_hull.compute(points, k, True)
            shape_ch = Polygon(boundary_points).buffer(0)
            shape = cascaded_union([shape, shape_ch])

        if type(shape) != Polygon:
            shape = max(shape, key=lambda s: s.area)

    elif k is not None:
        boundary_points = concave_hull.compute(points, k, True)
        shape = Polygon(boundary_points).buffer(0)
    else:
        raise ValueError('Either k or alpha needs to be set.')

    return shape


def fit_basic_shape(shape, max_error, given_angles=None):
    points = [Point(p) for p in shape.exterior.coords]
    convex_hull = ConvexHull(shape.exterior.coords)

    bounding_box = compute_bounding_box(
        np.array(shape.exterior.coords),
        convex_hull=convex_hull,
        given_angles=given_angles,
        max_error=max_error
    )
    dist_points_to_box = [
        bounding_box.exterior.distance(p) for p in points
    ]
    dist_box_to_points = [
        shape.exterior.distance(Point(p)) for
        p in bounding_box.exterior.coords
    ]
    bounding_box_dist = max(dist_points_to_box + dist_box_to_points)

    if max_error is not None and bounding_box_dist < max_error*2:
        return bounding_box, bounding_box_dist

    bounding_triangle = compute_bounding_triangle(
        np.array(shape.exterior.coords),
        convex_hull=convex_hull
    )
    dist_points_to_triangle = [
        bounding_triangle.exterior.distance(p) for p in points
    ]
    dist_triangle_to_points = [
        shape.exterior.distance(Point(p)) for
        p in bounding_triangle.exterior.coords
    ]
    bounding_triangle_dist = max(dist_points_to_triangle +
                                 dist_triangle_to_points)

    if max_error is not None and bounding_triangle_dist < max_error*2:
        return bounding_triangle, bounding_triangle_dist

    if bounding_box_dist < bounding_triangle_dist:
        return bounding_box, bounding_box_dist
    else:
        return bounding_triangle, bounding_triangle_dist
