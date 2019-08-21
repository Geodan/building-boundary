# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
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


def determine_non_fit_area(shape, basic_shape, max_error=None):
    diff = basic_shape - shape
    if max_error is not None:
        diff = diff.buffer(-max_error)
    return diff.area


def fit_basic_shape(shape, max_error=None, given_angles=None):
    convex_hull = ConvexHull(shape.exterior.coords)

    bounding_box = compute_bounding_box(
        np.array(shape.exterior.coords),
        convex_hull=convex_hull,
        given_angles=given_angles,
        max_error=max_error
    )

    bbox_non_fit_area = determine_non_fit_area(
        shape, bounding_box, max_error=max_error
    )
    if max_error is not None and bbox_non_fit_area <= 0:
        return bounding_box, True

    bounding_triangle = compute_bounding_triangle(
        np.array(shape.exterior.coords),
        convex_hull=convex_hull
    )

    tri_non_fit_area = determine_non_fit_area(
        shape, bounding_triangle, max_error=max_error
    )
    if max_error is not None and tri_non_fit_area <= 0:
        return bounding_triangle, True

    if bbox_non_fit_area < tri_non_fit_area:
        return bounding_box, False
    else:
        return bounding_triangle, False
