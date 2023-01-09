# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from shapely.ops import unary_union

import concave_hull

from .alpha_shape import compute_alpha_shape
from .bounding_box import compute_bounding_box
from .bounding_triangle import compute_bounding_triangle


def compute_shape(points, alpha=None, k=None):
    """
    Computes the shape of a set of points based on concave hulls.

    Parameters
    ----------
    points : (Mx2) array
        The coordinates of the points.
    alpha : float
        Set to compute the shape using an alpha shape using this
        chosen alpha. If both alpha and k are set both methods will be used and
        the resulting shapes merged to find the boundary points.
    k : int
        Set to compute the shape using a knn based concave hull
        algorithm using this amount of nearest neighbors. If both alpha and k
        are set both methods will be used and the resulting shapes merged to
        find the boundary points.

    Returns
    -------
    shape : polygon
        The computed shape of the points.
    """
    if alpha is not None:
        shape = compute_alpha_shape(points, alpha)

        if k is not None:
            boundary_points = concave_hull.compute(points, k, True)
            shape_ch = Polygon(boundary_points).buffer(0)
            shape = unary_union([shape, shape_ch])

        if type(shape) != Polygon:
            shape = max(shape.geoms, key=lambda s: s.area)

    elif k is not None:
        boundary_points = concave_hull.compute(points, k, True)
        shape = Polygon(boundary_points).buffer(0)
    else:
        raise ValueError('Either k or alpha needs to be set.')

    return shape


def determine_non_fit_area(shape, basic_shape, max_error=None):
    """
    Determines the area of the part of the basic shape that does not fit the
    given shape (i.e. what is left after differencing the two shapes,
    and optionally negative buffering with the max error).

    Parameters
    ----------
    shape : polygon
        The shape of the points.
    basic_shape : polygon
        The shape of the rectangle or triangle
    max_error : float, optional
        The maximum error (distance) a point may have to the shape.

    Returns
    -------
    area : float
        The area of the part of the basic shape that did not fit the
        given shape.
    """
    diff = basic_shape - shape
    if max_error is not None:
        diff = diff.buffer(-max_error)
    print('non fit area: {}'.format(diff.area))
    return diff.area


def fit_basic_shape(shape, max_error=None, given_angles=None):
    """
    Compares a shape to a rectangle and a triangle. If a max error is
    given it will return the shape and indicate that the basic shape fits
    well enough if that is the case.

    Parameters
    ----------
    shape : polygon
        The shape of the points.
    max_error : float, optional
        The maximum error (distance) a point may have to the shape.
    given_angles : list of float, optional
        If set, during the computation of the minimum area bounding box,
        the minimum area bounding box of these angles will be checked
        (instead of the angles of all edges of the convex hull).

    Returns
    -------
    basic_shape : polygon
        The polygon of the basic shape (rectangle or triangle) that fits the
        best.
    basic_shape_fits : bool
        If the found basic shape fits well enough within the given max error.
    """
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
