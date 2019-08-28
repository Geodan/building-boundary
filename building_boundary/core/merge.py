# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import bisect

import numpy as np

from .. import utils
from .segment import BoundarySegment
from .intersect import perpedicular_line_intersect


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
    ori_diff = np.fromiter((utils.angle.angle_difference(a1, a2) for
                            a1, a2 in utils.create_pairs(orientations)),
                           orientations.dtype)
    pivots_bool = ori_diff > angle
    pivots_idx = list(np.where(pivots_bool)[0] + 1)

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


def get_segments_between_pivots(segments, pivots):
    """
    Graps the segments between two pivots.

    Parameters
    ----------
    segments : list of BoundarySegment
        The segments.
    pivots : list of int
        The indices of the pivot points.

    Returns
    -------
    segments : list of int
        The indices of the segments between the pivots.
    """
    k, n = pivots
    if k < n:
        return list(range(k, n))
    else:  # edge case
        segments_pivots = list(range(k, len(segments)))
        segments_pivots.extend(range(0, n))
        return segments_pivots


def parallel_distance(segment1, segment2):
    """
    Computes the distance between two parallel segments.

    Parameters
    ----------
    segment1 : BoundarySegment
        A BoundarySegment.
    segment2 : BoundarySegment
        A subsequent BoundarySegment.

    Returns
    -------
    distance : float
        The distance between the two segments measured from the end point of
        segment 1 in a perpendicular line to the line of segment 2.
    """
    intersect = perpedicular_line_intersect(segment1, segment2)
    if len(intersect) > 0:
        return utils.geometry.distance(segment1.end_points[1], intersect)
    else:
        return float('inf')


def check_distance(segments, pivots, max_distance):
    """
    Checks if the distance between all subsequent parallel segments
    is larger than a given max, and inserts a pivot if this is the case.

    Parameters
    ----------
    segments : list of BoundarySegment
        The segments.
    pivots : list of int
        The indices of the pivot points.
    max_distance : float
        The maximum distance two parallel subsequent segments may be to be
        merged.

    Returns
    -------
    pivots : list of int
        The indices of the pivot points.
    """
    distances = []
    for i, pair in enumerate(utils.create_pairs(segments)):
        if (i + 1) % len(segments) not in pivots:
            distances.append(parallel_distance(pair[0], pair[1]))
        else:
            distances.append(float('nan'))
    too_far = np.where(np.array(distances) > max_distance)[0] + 1
    if len(too_far) > 0:
        too_far[-1] = too_far[-1] % len(segments)
        for x in too_far:
            bisect.insort_left(pivots, x)
    return pivots, distances


def merge_between_pivots(segments, start, end, max_error=None):
    """
    Merge the segments between two pivots.

    Parameters
    ----------
    segments : list of BoundarySegment
        The segments.
    start : int
        The first segment index.
    end : int
        Till which index segments should be merged.
    max_error : float
        The maximum error (distance) a point can have to a computed line.

    Returns
    -------
    merged_segment : BoundarySegment
        The segments merged.
    """
    if end == start + 1:
        return segments[start]
    else:
        points = get_points_between_pivots(
            segments,
            [start, end]
        )

        merged_segment = BoundarySegment(points)
        merge_segments = np.array(segments)[get_segments_between_pivots(
            segments, [start, end]
        )]
        longest_segment = max(merge_segments, key=lambda s: s.length)
        orientation = longest_segment.orientation
        merged_segment.regularize(orientation, max_error=max_error)
        return merged_segment


def merge_segments(segments, angle_epsilon=0.05,
                   max_distance=None, max_error=None):
    """
    Merges segments which are within a given angle of each
    other.

    Parameters
    ----------
    segments : list of BoundarySegment
        The segments.
    angle_epsilon : float
        The angle (in radians) difference within two angles are considered the
        same.
    max_distance : float
        If the distance between two parallel sequential segments (based on the
        angle epsilon) is lower than this value the segments get merged.
    max_error : float
        The maximum error (distance) a point can have to a computed line.

    Returns
    -------
    segments : list of BoundarySegment
        The new set of segments
    """
    orientations = np.array([s.orientation for s in segments])
    pivots = find_pivots(orientations, angle_epsilon)

    if max_distance is not None:
        pivots, distances = check_distance(segments, pivots, max_distance)

    while True:
        new_segments = []
        try:
            for pivot_segment in utils.create_pairs(pivots):
                new_segment = merge_between_pivots(
                    segments, pivot_segment[0], pivot_segment[1], max_error
                )
                new_segments.append(new_segment)
            break
        except utils.error.ThresholdError:
            segments_idx = get_segments_between_pivots(segments, pivot_segment)
            new_pivot_1 = segments_idx[
                np.nanargmax(np.array(distances)[segments_idx])
            ]
            new_pivot_2 = new_pivot_1 + 1
            if new_pivot_1 not in pivots:
                bisect.insort_left(pivots, new_pivot_1)
            if new_pivot_2 not in pivots:
                bisect.insort_left(pivots, new_pivot_2)

    return new_segments
