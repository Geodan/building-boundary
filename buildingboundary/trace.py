# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import numpy as np
from shapely.geometry import Polygon
from shapely.wkt import loads
from shapely.ops import cascaded_union

import concave_hull

from .components.alphashape import compute_alpha_shape
from .components.boundingbox import compute_bounding_box
from .components.segment import BoundarySegment
from .components.segmentation import boundary_segmentation
from .components.merge import merge_offset_lines
from .components.intersect import compute_intersections
from .components.regularize import (get_primary_orientations,
                                    regularize_segments,
                                    geometry_orientations)
from .utils.angle import perpendicular


def trace_boundary(points, max_error, alpha=None, k=None,
                   min_area=0, max_rectangularity=0.97,
                   merge_angle=None, merge_distance=None, num_points=None,
                   primary_orientations=None, perp_dist_weight=3,
                   max_error_invalid=None, inflate=False,
                   footprint_geom=None):
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
        order = 'cw'
        shape = compute_alpha_shape(points, alpha)

        if k is not None:
            boundary_points = concave_hull.compute(points, k, True)
            shape_ch = Polygon(boundary_points).buffer(0)
            shape = cascaded_union([shape, shape_ch])

        if type(shape) != Polygon:
            shape = max(shape, key=lambda s: s.area)

            boundary_points = np.array(shape.exterior.coords)
    elif k is not None:
        order = 'ccw'
        boundary_points = concave_hull.compute(points, k, True)
        shape = Polygon(boundary_points).buffer(0)
    else:
        raise ValueError('Either k or alpha needs to be set.')

    if primary_orientations is None and footprint_geom is not None:
        primary_orientations = geometry_orientations(loads(footprint_geom))

    bounding_box = compute_bounding_box(boundary_points,
                                        given_angles=primary_orientations,
                                        max_error=max_error_invalid)

    if shape.area < min_area:
        return np.array(bounding_box.exterior.coords)

    rectangularity = shape.area / bounding_box.area
    if rectangularity > max_rectangularity:
        return np.array(bounding_box.exterior.coords)

    segments = boundary_segmentation(boundary_points, max_error)

    if len(segments) in [0, 1, 2]:
        return np.array(bounding_box.exterior.coords)

    boundary_segments = [BoundarySegment(s) for s in segments]
    for s in boundary_segments:
        s.fit_line(method='TLS')

    if primary_orientations is None or len(primary_orientations) == 0:
        primary_orientations = get_primary_orientations(boundary_segments,
                                                        num_points)

    if len(primary_orientations) == 1:
        primary_orientations.append(perpendicular(primary_orientations[0]))

    boundary_segments = regularize_segments(boundary_segments,
                                            primary_orientations,
                                            max_error=max_error_invalid)

    if inflate:
        for s in boundary_segments:
            s.inflate(order=order)

    vertices = compute_intersections(boundary_segments,
                                     perp_dist_weight=perp_dist_weight)

    vertices = merge_offset_lines(vertices,
                                  merge_angle,
                                  merge_distance)

    if not Polygon(vertices).is_valid:
        return np.array(bounding_box.exterior.coords)

    return vertices
