# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import math
import numpy as np
from scipy.spatial import ConvexHull

from ..utils.error import ThresholdError
from ..utils.angle import angle_difference, min_angle_difference
from .segment import BoundarySegment


def create_segments(points):
    return zip(points, np.roll(points, -1))


def convex_fit(points, boundary_segments, max_error):
    hull = ConvexHull(points)
    hull.vertices.sort()
    segments = create_segments(hull.vertices)

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
        except:  # ThresholdError:
            convex_fit(s_points, boundary_segments, max_error=max_error)


def merge_segments(segments, merge_angle):
    prev_segments = segments.copy()
    n_segments = len(segments)
    n_prev_segments = 0

    merged_segments = []
    iteration = 0

    while n_segments != n_prev_segments:
        n_prev_segments = len(prev_segments)
        new_segments = []

        merged_segments.append([])

        orientations = np.array([s.orientation for s in prev_segments])
        ori_diff = np.fromiter((angle_difference(a1, a2) for
                                a1, a2 in zip(orientations,
                                              np.roll(orientations, -1))),
                               orientations.dtype)
        pivots_bool = ori_diff > merge_angle
        pivots_idx = list(np.where(pivots_bool == True)[0] + 1)

        # edge case
        if pivots_idx[-1] > (len(prev_segments)-1):
            pivots_idx[-1] = 0

        for i, (k, n) in enumerate(zip(pivots_idx, np.roll(pivots_idx, -1))):
            points = [prev_segments[k].points[0]]
            if k < n:
                for s in prev_segments[k:n]:
                    points.extend(s.points[1:])
            else:  # edge case
                for s in prev_segments[k:]:
                    points.extend(s.points[1:])
                for s in prev_segments[:n]:
                    points.extend(s.points[1:])

            merged_segment = BoundarySegment(np.array(points))
            merged_segment.fit_line(method='TLS')
            new_segments.append(merged_segment)

            merged_segments[-1].append(list(range(k, n)))

        n_segments = len(new_segments)
        prev_segments = new_segments

        iteration += 1

    # track merged segments
    if iteration > 2:
        prev_merged_segments = merged_segments[-2]
        for next_merged_segments in merged_segments[::-1][2:]:
            new_merged_segments = []
            for s in prev_merged_segments:
                new_segment = []
                for j in s:
                    new_segment.extend(next_merged_segments[j])
                new_merged_segments.append(new_segment)
            prev_merged_segments = new_merged_segments.copy()
        merged_segments = new_merged_segments
    else:
        merged_segments = merged_segments[0]

    return new_segments, merged_segments


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

            # remove if close to 90 degree angle
            if (angle > math.radians(80) and angle < math.radians(100)):
                to_remove.append(i)

    new_segments = [s for i, s in enumerate(segments) if i not in to_remove]

    return new_segments, to_remove
