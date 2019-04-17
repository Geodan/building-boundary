# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import numpy as np
from shapely.geometry import Polygon

import concave_hull

from .components.alphashape import compute_alpha_shape
from .components.boundingbox import compute_bounding_box
from .components.segment import BoundarySegment
from .components.segmentation import boundary_segmentation
from .components.merge import merge_segments, flatten_merge_history
from .components.intersect import compute_intersections
from .components.regularize import (get_primary_orientations,
                                    regularize_and_merge)
from .components.assess import check_error, restore
from .utils.angle import perpendicular


def trace_boundary(points, max_error, merge_angle, alpha=None,
                   k=None, min_area=0, max_rectangularity=0.97,
                   max_merge_distance=None, num_points=None,
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
    if alpha is not None:
        shape = compute_alpha_shape(points, alpha)
        if type(shape) == Polygon:
            boundary_points = np.array(shape.exterior.coords)
        else:
            largest_polygon = max(shape, key=lambda s: s.area)
            boundary_points = np.array(largest_polygon.exterior.coords)
    elif k is not None:
        boundary_points = concave_hull.compute(points, k, True)
        shape = Polygon(boundary_points)
    else:
        raise ValueError('Either k or alpha needs to be set.')

    bounding_box = compute_bounding_box(boundary_points,
                                        given_angles=primary_orientations)

    if shape.area < min_area:
        return bounding_box

    rectangularity = shape.area / bounding_box.area
    if rectangularity > max_rectangularity:
        return np.array(bounding_box.exterior.coords)

    segments = boundary_segmentation(boundary_points, max_error)
    boundary_segments = [BoundarySegment(s) for s in segments]
    for s in boundary_segments:
        s.fit_line(method='TLS')

    original_segments = boundary_segments.copy()

    boundary_segments, merge_history_1 = merge_segments(boundary_segments,
                                                        merge_angle,
                                                        max_merge_distance)

    if primary_orientations is None or len(primary_orientations) == 0:
        primary_orientations = get_primary_orientations(boundary_segments,
                                                        num_points)
    elif len(primary_orientations) == 1:
        primary_orientations.append(perpendicular(primary_orientations[0]))

    boundary_segments, merge_history_2 = regularize_and_merge(boundary_segments,
                                                              primary_orientations,
                                                              merge_angle,
                                                              max_error,
                                                              max_merge_distance)

    invalid_segments = check_error(boundary_segments, max_error)
    if len(invalid_segments) > 0:
        merge_history = merge_history_1 + merge_history_2
        merged_segments = flatten_merge_history(merge_history)
        boundary_segments = restore(boundary_segments, original_segments,
                                    invalid_segments, merged_segments)

    if inflate:
        for s in boundary_segments:
            s.inflate(order='cw')


    vertices = compute_intersections(boundary_segments)

    return vertices
