# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import math
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
    prev_segments = segments.copy()
    n_segments = len(segments)
    n_prev_segments = 0

    while n_segments != n_prev_segments:
        n_prev_segments = len(prev_segments)

        orientations = np.array([s.orientation for s in prev_segments])
    ori_diff = np.fromiter((angle_difference(a1, a2) for
                            a1, a2 in zip(orientations,
                                          np.roll(orientations, -1))),
                           orientations.dtype)
    pivots_bool = ori_diff > merge_angle
        pivots_idx = np.array([0] + list(np.where(pivots_bool == True)[0] + 1))
    new_segments = []

    for i, j in zip(pivots_idx[:-1], np.roll(pivots_idx, -1)[:-1]):
            points = [prev_segments[i].points[0]]
            for s in prev_segments[i:j]:
                points.extend(s.points[1:])
        merged_segment = BoundarySegment(np.array(points))
        merged_segment.fit_line(method='TLS')
        new_segments.append(merged_segment)

        n_segments = len(new_segments)
        prev_segments = new_segments

    return new_segments
def remove_small_corners(segments, n_points=2):
    to_remove = []
    n_segments = len(segments)

    for i in range(n_segments):
        s = segments[i]
        if len(s.points) <= n_points:

            # Edge cases for first and last segment
            if i == 0:
                angle = angle_difference(segments[n_segments-1].orientation,
                                         segments[1].orientation)
            elif i == n_segments-1:
                angle = angle_difference(segments[i-1].orientation,
                                         segments[0].orientation)
            else:
                angle = angle_difference(segments[i-1].orientation,
                                         segments[i+1].orientation)

            if (angle > math.radians(80) and angle < math.radians(100)):
                to_remove.append(i)

    new_segments = [s for i, s in enumerate(segments) if i not in to_remove]

    return new_segments
