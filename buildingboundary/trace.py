# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import numpy as np
from shapely.geometry import Polygon

import concave_hull

from .components.alphashape import compute_alpha_shape
from .components.intersect import compute_intersections
from .components.regularize import (get_primary_orientations,
                                    regularize_and_merge)
from .components.assess import check_error, restore
from .utils.angle import perpendicular


def trace_boundary(points, max_error, merge_angle, k=None, alpha=None,
                   num_points=float('inf'),
                   primary_orientations=None, inflate=False):
    """
    Trace the boundary of a set of 2D points.

    Parameters
    ----------
    points : (Mx2) array
        The coordinates of the points
    k : int
        The amount of nearest neighbors used in the concave hull algorithm
    max_error : float
        The maximum average error (distance) the points make with the fitted
        line
    merge_angle : float
        The angle (in radians) difference within two segments will be merged
    num_points : int, optional
        The number of points a segment needs to be supported by to be
        considered a primary orientation. Will be ignored if primary
        orientations are set manually.
    max_intersect_distance : float, optional
        The maximum distance an found intersection can be from the segments.
    alignment : float, optional
        If set segments will be aligned (their intercept be set equal by
        averaging) if the difference between their x-axis intercepts is within
        this number.
    primary_orientations : list of floats, optional
        The desired primary orientations (in radians) of the boundary. If set
        manually here these orientations will not be computed.
    inflate : bool
        If set to true the fit lines will be moved to the furthest outside
        point.

    Returns
    -------
    : (Mx2) array
        The vertices of the computed boundary line
    """
    if k is not None:
    boundary_points = concave_hull.compute(points, k, True)
        shape = Polygon(boundary_points)
    elif alpha is not None:
        shape = compute_alpha_shape(points, 0.5)
        boundary_points = np.array(shape.exterior.coords)
    else:
        raise ValueError('Either k or alpha needs to be set.')

    boundary_segments = []
    convex_fit(boundary_points, boundary_segments, max_error=max_error)
    # original_segments = boundary_segments.copy()

    boundary_segments, merge_history = merge_segments(boundary_segments,
                                                      merge_angle)

    # boundary_segments, removed_segments = remove_small_corners(boundary_segments)

    if len(boundary_segments) in [0, 1, 2]:
        return []
    elif len(boundary_segments) == 3:
        vertices = compute_intersections(boundary_segments)
        return vertices

    if primary_orientations is None or len(primary_orientations) == 0:
        primary_orientations = get_primary_orientations(boundary_segments,
                                                        num_points)
    elif len(primary_orientations) == 1:
        primary_orientations.append(perpendicular(primary_orientations[0]))

    boundary_segments = regularize_and_merge(boundary_segments, primary_orientations,
                                             merge_angle, max_error)

    # invalid_segments = check_error(boundary_segments, max_error*1.5)
    # boundary_segments = restore(boundary_segments, original_segments, invalid_segments,
    #                             merged_segments, removed_segments)

    if inflate:
        for s in boundary_segments:
            s.inflate()

    vertices = compute_intersections(boundary_segments)

    return vertices
