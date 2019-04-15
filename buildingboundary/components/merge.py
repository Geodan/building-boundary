# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import math
from bisect import insort_left

import numpy as np

from ..utils.angle import angle_difference
from ..utils.utils import create_segments, distance
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
    pivots_idx = list(np.where(pivots_bool == True)[0] + 1)

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


def check_distance(segments, pivots, max_distance):
    distances = np.array([distance(pair[0].end_points[1],
                                   pair[1].end_points[0])
                         for pair in create_segments(segments)])
    too_far = np.where(distances > max_distance)[0] + 1
    too_far[-1] = 0 if too_far[-1] > len(segments) - 1 else too_far[-1]
    for x in too_far:
        insort_left(pivots, x)
    pivots = list(set(pivots))
    return pivots


def merge_segments(segments, merge_angle, max_distance=float('inf')):
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
        if max_distance != float('inf'):
            pivots = check_distance(prev_segments, pivots, max_distance)

        for pivot_segment in create_segments(pivots):
            points = get_points_between_pivots(prev_segments, pivot_segment)

            merged_segment = BoundarySegment(points)
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
