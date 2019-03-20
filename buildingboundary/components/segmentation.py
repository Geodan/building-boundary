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


def create_segments(iterable):
    """
    Creates pairs in sequence between the elements of the
    iterable.

    Parameters
    ----------
    iterable : list
        The interable to create pairs of.

    Returns
    -------
    segments : zip
        Iterator which contains pairs of points that define
        the segments.

    Examples
    --------
    >>> list(create_segments([3, 5, 1, 9, 8]))
    [(3, 5), (5, 1), (1, 9), (9, 8), (8, 3)]
    """
    return zip(iterable, np.roll(iterable, -1))


def convex_fit(points, boundary_segments, max_error):
    """
    Recursive function which segments a set of points into
    linear segments.

    Parameters
    ----------
    points : (Mx2) array
        The coordinates of the points
    boundary_segments : list of BoundarySegment
        The accepted boundary segments. Initiate using a variable
        containing an empty list.
    max_error : float or int
        The maximum error (average distance points to line) the
        fitted line is allowed to have.
    """
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
        except ThresholdError:
            convex_fit(s_points, boundary_segments, max_error=max_error)


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
    pivots_idx = list(np.where(pivots_bool == True)[0] + 1)

    # edge case
    if pivots_idx[-1] > (len(orientations)-1):
        pivots_idx[-1] = 0

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
    points : list of array

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


def get_merged_segments(pivot_segment, n_segments):
    """
    """
    merged_segments = []
    if pivot_segment[1] > pivot_segment[0]:
        merged_segments += list(range(pivot_segment[0],
                                      pivot_segment[1]))
    else:
        merged_segments += list(range(pivot_segment[0], n_segments))
        merged_segments += list(range(0, pivot_segment[1]))
    return merged_segments


def merge_segments(segments, merge_angle):
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

    merge_history = []

    while n_segments != n_prev_segments:
        n_prev_segments = len(prev_segments)
        new_segments = []

        merge_history.append([])

        orientations = np.array([s.orientation for s in prev_segments])
        pivots = find_pivots(orientations, merge_angle)

        for pivot_segment in create_segments(pivots):
            points = get_points_between_pivots(prev_segments, pivot_segment)

            merged_segment = BoundarySegment(np.array(points))
            merged_segment.fit_line(method='TLS')
            new_segments.append(merged_segment)

            merged_segments = get_merged_segments(pivot_segment,
                                                  len(prev_segments))

            merge_history[-1].append(merged_segments)

        n_segments = len(new_segments)
        prev_segments = new_segments

    return new_segments, merge_history


def flatten_merge_history(merge_history):
    """
    Flattens a series of lists which contain which segments have
    been merged in each iteration.

    Parameters
    ----------
    merge_history : list of list of list of int
        The merge history

    Returns
    -------
    merged_segments : list of list of int
    """
    if len(merge_history) > 2:
        prev_merged_segments = merge_history[-2]
        for next_merged_segments in merge_history[::-1][2:]:
            new_merged_segments = []
            for s in prev_merged_segments:
                new_segment = []
                for j in s:
                    new_segment.extend(next_merged_segments[j])
                new_merged_segments.append(new_segment)
            prev_merged_segments = new_merged_segments.copy()
        merge_history_flat = new_merged_segments
    else:
        merge_history_flat = merge_history[0]

    return merge_history_flat


def remove_small_corners(segments, n_points=2, angle_epsilon=0.1745):
    """
    Removes small segments which are between two segments that
    are about 90 degrees from each other.

    Parameters
    ----------
    segments : list of BoundarySegment
        The list of boundary segments.
    n_points : int, optional
        The maximum number of points for a segment to be
        considered a small segment.
    angle_epsilon : float, optional
        Angles will be considered equal if the difference is within
        this value (in radians).

    Returns
    -------
    segments : list of BoundarySegment
        The new boundary segments.
    removed : list of int
        The indices of the removed boundary segments.
    """
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
            if (angle > (math.pi/2 - angle_epsilon) and
                    angle < (math.pi/2 + angle_epsilon)):
                to_remove.append(i)

    new_segments = [s for i, s in enumerate(segments) if i not in to_remove]

    return new_segments, to_remove
