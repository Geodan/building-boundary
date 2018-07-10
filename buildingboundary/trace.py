# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import concave_hull

from .components.segmentation import convex_fit, merge_segments
from .components.intersect import compute_intersections
from .components.regularize import compute_primary_orientations, regularize_lines
from .components.align import align_by_intersept


def trace_boundary(points, k, max_error, merge_angle, num_points=float('inf'),
                   alignment=0, inflate=False):
    """
    """
    boundary_points = concave_hull.compute(points, k, True)

    boundary_segments = []
    convex_fit(boundary_points, boundary_segments, max_error=max_error)

    boundary_segments = merge_segments(boundary_segments, merge_angle)

    if len(boundary_segments) in [0, 1, 2]:
        return []
    elif len(boundary_segments) == 3:
        vertices = compute_intersections(boundary_segments)
        return vertices

    primary_orientations = compute_primary_orientations(boundary_segments,
                                                        num_points)

    boundary_segments = regularize_lines(boundary_segments,
                                         primary_orientations,
                                         merge_angle)

    if alignment != 0:
        align_by_intersept(boundary_segments, alignment)

    vertices = compute_intersections(boundary_segments)

    if inflate:
        for s in boundary_segments:
            s.inflate()
        vertices = compute_intersections(boundary_segments)

    return vertices
