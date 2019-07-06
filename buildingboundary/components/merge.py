# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import bisect

import numpy as np

from ..utils.angle import angle_difference
from ..utils.error import ThresholdError
from ..utils import create_segments, distance
from .segment import BoundarySegment


def find_pivots(orientations, angle):
    """
    Finds the indices of where the difference in orientation is
    larger than the given angle.

    Parameters
    ----------
    orientations : list of float
        The sequence of orientations
    angle : float or int
        The difference in angle at which a point will be considered a
        pivot.

    Returns
    -------
    pivot_indices : list of int

    """
    ori_diff = np.fromiter((angle_difference(a1, a2) for
                            a1, a2 in create_segments(orientations)),
                           orientations.dtype)
    pivots_bool = ori_diff > angle
    pivots_idx = list(np.where(pivots_bool == True)[0] + 1)  # noqa

    # edge case
    if pivots_idx[-1] > (len(orientations)-1):
        del pivots_idx[-1]
        pivots_idx[0:0] = [0]

    return pivots_idx


def get_points_between_pivots(segments, pivots):
    """
    Returns the points between two pivot points

    Parameters
    ----------
    segments : list of BoundarySegment
        The segments.
    pivots : list of int
        The indices of the pivot points.

    Returns
    -------
    points : (Mx2) array

    """
    k, n = pivots
    points = [segments[k].points[0]]
    if k < n:
        for s in segments[k:n]:
            points.extend(s.points[1:])
    else:  # edge case
        for s in segments[k:]:
            points.extend(s.points[1:])
        for s in segments[:n]:
            points.extend(s.points[1:])

    return np.array(points)


def check_distance(segments, pivots, max_distance):
    distances = np.array([distance(pair[0].end_points[1],
                                   pair[1].end_points[0])
                         for pair in create_segments(segments)])
    too_far = np.where(distances > max_distance)[0] + 1
    if len(too_far) > 0:
        too_far[-1] = 0 if too_far[-1] > len(segments) - 1 else too_far[-1]
        for x in too_far:
            bisect.insort_left(pivots, x)
        pivots = list(set(pivots))
    return pivots


def get_segments_between_pivots(segments, pivots):
    k, n = pivots
    if k < n:
        return [s for s in segments[k:n]]
    else:  # edge case
        segments_pivots = []
        for s in segments[k:]:
            segments_pivots.append(s)
        for s in segments[:n]:
            segments_pivots.append(s)
        return segments_pivots


def merge_segments(segments, merge_angle, max_distance=None, max_error=None):
    """
    Merges segments which are within a given angle of each
    other.

    Parameters
    ----------
    segments : list of BoundarySegment
        The segments.
    merge_angle : float or int


    Returns
    -------
    segments : list of BoundarySegment

    merge_history : list of list of int

    """
    prev_segments = segments.copy()
    n_segments = len(segments)
    n_prev_segments = 0

    while n_segments != n_prev_segments:
        n_prev_segments = len(prev_segments)
        new_segments = []

        orientations = np.array([s.orientation for s in prev_segments])
        pivots = find_pivots(orientations, merge_angle)
        if max_distance is not None:
            pivots = check_distance(prev_segments, pivots, max_distance)

        for pivot_segment in create_segments(pivots):
            try:
                points = get_points_between_pivots(
                    prev_segments,
                    pivot_segment
                )

                merged_segment = BoundarySegment(points)
                merged_segment.fit_line(method='TLS', max_error=max_error)
                new_segments.append(merged_segment)
            except ThresholdError:
                new_segments.extend(
                    get_segments_between_pivots(prev_segments, pivot_segment)
                )

        n_segments = len(new_segments)
        prev_segments = new_segments

    return new_segments
