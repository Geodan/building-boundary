# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import numpy as np
from scipy.spatial import ConvexHull

from ..utils.error import ThresholdError
from ..utils.angle import angle_difference
from .segment import BoundarySegment


def convex_fit(points, boundary_segments, max_error):
    hull = ConvexHull(points)
    hull.vertices.sort()
    segments = zip(hull.vertices, np.roll(hull.vertices, -1))

    for s in segments:
        if s[1]+1 == len(hull.points) or s[1] == 0:
            s_points = hull.points[s[0]:]
        else:
            s_points = hull.points[s[0]:s[1]+1]

        if len(s_points) == 1:
            continue

        try:
            segment = BoundarySegment(s_points)
            segment.fit_line(method='TLS', max_error=max_error)
            boundary_segments.append(segment)
        except ThresholdError:
            convex_fit(s_points, boundary_segments, max_error=max_error)


def merge_segments(segments, merge_angle):
    orientations = np.array([s.orientation for s in segments])
    ori_diff = np.fromiter((angle_difference(a1, a2) for
                            a1, a2 in zip(orientations,
                                          np.roll(orientations, -1))),
                           orientations.dtype)
    pivots_bool = ori_diff > merge_angle
    pivots_idx = [0] + list(np.where(pivots_bool == True)[0] + 1)
    new_segments = []

    for i, j in zip(pivots_idx[:-1], np.roll(pivots_idx, -1)[:-1]):
        points = []
        for s in segments[i:j]:
            points.extend(s.points)
        merged_segment = BoundarySegment(np.array(points))
        merged_segment.fit_line(method='TLS')
        new_segments.append(merged_segment)

    return new_segments
