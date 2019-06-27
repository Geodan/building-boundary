# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import math
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


def line_angle(line):
    dx, dy = np.diff(line, axis=0)[0]
    angle = math.atan2(dy, dx)
    return angle


def has_offset_line(line_idx, lines, merge_angle, max_distance):
    num_lines = len(lines)

    l1 = lines[line_idx]
    l2 = lines[(line_idx + 1) % num_lines]
    l3 = lines[(line_idx + 2) % num_lines]

    if distance(*l2) < max_distance:
        a1 = line_angle(l1)
        a2 = line_angle(l3)

        return angle_difference(a1, a2) < merge_angle
    else:
        return False


def subsequent_offset_lines(line_idx, lines, merge_angle,
                            max_distance, merged_lines):
    num_lines = len(lines)

    offset = 2
    while True:
        next_line_idx = line_idx + offset

        if next_line_idx >= num_lines:
            next_line_idx = next_line_idx % num_lines
            if (next_line_idx in merged_lines or
                    next_line_idx + 1 in merged_lines):
                offset -= 2
                break

        if has_offset_line(next_line_idx, lines, merge_angle, max_distance):
            offset += 2
        else:
            offset -= 2
            break
    return offset


def compute_new_vertex(line_1, line_2):
    """
    https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_points_on_each_line
    """
    [[x1, y1], [x2, y2]] = line_1
    [[x3, y3], [x4, y4]] = line_2
    px = (((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) /
          ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)))
    py = (((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) /
          ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)))
    return (px, py)


def merge_offset_lines(vertices, merge_angle, max_distance):
    new_vertices = []
    merged_lines = []

    lines = list(create_segments(vertices))

    num_lines = len(lines)
    line_idx = 0
    while line_idx < num_lines:
        if has_offset_line(line_idx, lines, merge_angle, max_distance):
            offset = 3
            offset += subsequent_offset_lines(line_idx, lines, merge_angle,
                                              max_distance, merged_lines)

            line_1_idx = (line_idx - 1) % num_lines
            line_2_idx = (line_idx + offset - 1) % num_lines
            line_1 = lines[line_1_idx]
            line_2 = lines[line_2_idx]

            if line_2_idx > line_1_idx:
                merged_lines.extend(
                    [i for i in range(line_1_idx, line_2_idx + 1)]
                )
            else:
                merged_lines.extend([i for i in range(line_1_idx, num_lines)])
                merged_lines.extend([i for i in range(0, line_2_idx + 1)])

            new_vertex = compute_new_vertex(line_1, line_2)
            new_vertices.append(new_vertex)

            line_idx += offset
        else:
            new_vertices.append(vertices[line_idx])
            line_idx += 1

    # if merge found at edge case, remove first found vertex/vertices
    new_vertices = new_vertices[line_idx - num_lines:]

    return np.array(new_vertices)
