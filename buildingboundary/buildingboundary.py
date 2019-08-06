# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import numpy as np
from shapely.geometry import Polygon, Point
from shapely.ops import cascaded_union

import concave_hull

from .shapes.alphashape import compute_alpha_shape
from .shapes.boundingbox import compute_bounding_box
from .components.segment import BoundarySegment
from .components.segmentation import boundary_segmentation
from .components.merge import merge_segments
from .components.intersect import compute_intersections
from .components.regularize import (get_primary_orientations,
                                    regularize_segments)
from .components.inflate import inflate_polygon
from . import utils


def trace_boundary(points, max_error, alpha=None, k=None,
                   num_points=None, angle_epsilon=0.05, merge_distance=None,
                   primary_orientations=None, perp_dist_weight=3,
                   max_error_invalid=None, inflate=False):
    """
    Trace the boundary of a set of 2D points.

    Parameters
    ----------
    points : (Mx2) array
        The coordinates of the points.
    max_error : float
        The maximum error (distance) the points make with the fitted
        line.

    k : int
        The amount of nearest neighbors used in the concave hull algorithm

    merge_angle : float
        The angle (in radians) difference within two segments can be merged

    num_points : int, optional
        The number of points a segment needs to be supported by to be
        considered a primary orientation. Will be ignored if primary
        orientations are set manually.


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

        if k is not None:
            boundary_points = concave_hull.compute(points, k, True)
            shape_ch = Polygon(boundary_points).buffer(0)
            shape = cascaded_union([shape, shape_ch])

        if type(shape) != Polygon:
            shape = max(shape, key=lambda s: s.area)

        boundary_points = np.array(shape.exterior.coords)
    elif k is not None:
        boundary_points = concave_hull.compute(points, k, True)
        shape = Polygon(boundary_points).buffer(0)
    else:
        raise ValueError('Either k or alpha needs to be set.')

    bounding_box = compute_bounding_box(boundary_points,
                                        given_angles=primary_orientations,
                                        max_error=max_error_invalid)

    distances = [bounding_box.exterior.distance(Point(p)) for
                 p in boundary_points]
    if max(distances) < max_error_invalid:
        return np.array(bounding_box.exterior.coords)

    segments = boundary_segmentation(boundary_points, max_error)

    if len(segments) in [0, 1, 2]:
        return np.array(bounding_box.exterior.coords)

    boundary_segments = [BoundarySegment(s) for s in segments]

    if primary_orientations is None or len(primary_orientations) == 0:
        primary_orientations = get_primary_orientations(boundary_segments,
                                                        num_points)

    if len(primary_orientations) == 1:
        primary_orientations.append(
            utils.angle.perpendicular(primary_orientations[0])
        )

    boundary_segments = regularize_segments(boundary_segments,
                                            primary_orientations,
                                            max_error=max_error_invalid)

    boundary_segments = merge_segments(boundary_segments,
                                       angle_epsilon=angle_epsilon,
                                       max_distance=merge_distance,
                                       max_error=max_error_invalid)

    vertices = compute_intersections(boundary_segments,
                                     perp_dist_weight=perp_dist_weight)

    if inflate:
        vertices = inflate_polygon(vertices, boundary_points)

    if not Polygon(vertices).is_valid:
        return np.array(bounding_box.exterior.coords)

    return vertices
