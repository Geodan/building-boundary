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
from .core.segment import BoundarySegment
from .core.segmentation import boundary_segmentation
from .core.merge import merge_segments
from .core.intersect import compute_intersections
from .core.regularize import get_primary_orientations, regularize_segments
from .core.inflate import inflate_polygon
from . import utils


def trace_boundary(points, ransac_threshold, max_error=None, alpha=None,
                   k=None, num_points=None, angle_epsilon=0.05,
                   merge_distance=None, primary_orientations=None,
                   perp_dist_weight=3, inflate=False):
    """
    Trace the boundary of a set of 2D points.

    Parameters
    ----------
    points : (Mx2) array
        The coordinates of the points.
    ransac_threshold : float
        Maximum distance for a data point to be classified as an inlier during
        the RANSAC line fitting.
    max_error : float
        The maximum error (distance) a point can have to a computed line.
    alpha : float
        Set to determine the boundary points using an alpha shape using this
        chosen alpha. If both alpha and k are set both methods will be used and
        the resulting shapes merged to find the boundary points.
    k : int
        Set to determine the boundary points using a knn based concave hull
        algorithm using this amount of nearest neighbors. If both alpha and k
        are set both methods will be used and the resulting shapes merged to
        find the boundary points.
    num_points : int, optional
        The number of points a segment needs to be supported by to be
        considered a primary orientation. Will be ignored if primary
        orientations are set manually.
    angle_epsilon : float
        The angle (in radians) difference within two angles are considered the
        same. Used to merge segments.
    merge_distance : float
        If the distance between two parallel sequential segments (based on the
        angle epsilon) is lower than this value the segments get merged.
    primary_orientations : list of floats, optional
        The desired primary orientations (in radians) of the boundary. If set
        manually here these orientations will not be computed.
    perp_dist_weight : float
        Used during the computation of the intersections between the segments.
        If the distance between the intersection of two segments and the
        segments is more than `perp_dist_weight` times the distance between the
        intersection of the perpendicular line at the end of the line segment
        and the segments, the perpendicular intersection will be used instead.
    inflate : bool
        If set to true the fit lines will be moved to the furthest outside
        point.

    Returns
    -------
    vertices : (Mx2) array
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
                                        max_error=max_error)

    distances = [bounding_box.exterior.distance(Point(p)) for
                 p in boundary_points]
    if max_error is not None and max(distances) < max_error:
        return np.array(bounding_box.exterior.coords)

    segments = boundary_segmentation(boundary_points, ransac_threshold)

    if len(segments) in [0, 1, 2]:
        return np.array(bounding_box.exterior.coords)

    boundary_segments = [BoundarySegment(s) for s in segments]

    if primary_orientations is None or len(primary_orientations) == 0:
        primary_orientations = get_primary_orientations(
            boundary_segments,
            num_points,
            angle_epsilon=angle_epsilon
        )

    if len(primary_orientations) == 1:
        primary_orientations.append(
            utils.angle.perpendicular(primary_orientations[0])
        )

    boundary_segments = regularize_segments(boundary_segments,
                                            primary_orientations,
                                            max_error=max_error)

    boundary_segments = merge_segments(boundary_segments,
                                       angle_epsilon=angle_epsilon,
                                       max_distance=merge_distance,
                                       max_error=max_error)

    vertices = compute_intersections(boundary_segments,
                                     perp_dist_weight=perp_dist_weight)

    if inflate:
        remaining_points = boundary_segments[0].points
        for s in boundary_segments[1:]:
            remaining_points = np.vstack((remaining_points, s.points))
        vertices = inflate_polygon(vertices, remaining_points)

    if not Polygon(vertices).is_valid:
        return np.array(bounding_box.exterior.coords)

    return vertices
