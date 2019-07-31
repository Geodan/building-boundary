# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import math

import numpy as np

from ..utils.angle import angle_difference
from ..utils import create_segments, distance


def line_angle(line):
    """
    Computes the angle of a line defined by two points.

    Parameters
    ----------
    line : (2x2) array
        The coordinates of the two points defining the line.

    Returns
    -------
        Angle of the line in radians from
        0 to pi (east to west counterclockwise)
        0 to -pi (east to west clockwise)
    """
    dx, dy = np.diff(line, axis=0)[0]
    angle = math.atan2(dy, dx)
    return angle


def has_offset_line(line_idx, lines, merge_angle, max_distance):
    """
    Determines if a line has a subsequent line that is parallel and offset
    by a small distance.

    Parameters
    ----------
    line_idx : int
        The index of the line.
    lines : list of (2x2) array
        The lines defined by the coordinates two points.
    merge_angle : float
        The angle difference two lines may have to be considered parallel.
    max_distance : float
        The maximum distance the offset can be for the lines to be merged.

    Returns
    -------
     : bool
        If the line has a subsequent parallel offset line.
    """
    num_lines = len(lines)

    l2i = (line_idx + 1) % num_lines
    l3i = (line_idx + 2) % num_lines

    if line_idx == l2i or line_idx == l3i:
        return False

    l1 = lines[line_idx]
    l2 = lines[l2i]
    l3 = lines[l3i]

    if distance(*l2) < max_distance:
        a1 = line_angle(l1)
        a2 = line_angle(l3)

        return angle_difference(a1, a2) < merge_angle
    else:
        return False


def subsequent_offset_lines(line_idx, lines, merge_angle,
                            max_distance, merged_lines):
    """
    Determines if subsequent lines are also parallel and offset by a small
    distance.

    Parameters
    ----------
    line_idx : int
        The index of the line.
    lines : list of (2x2) array
        The lines defined by the coordinates two points.
    merge_angle : float
        The angle difference two lines may have to be considered parallel.
    max_distance : float
        The maximum distance the offset can be for the lines to be merged.
    merged_lines : list of int
        The indices of the lines that have already been merged.

    Returns
    -------
    offset : int
        How many lines can be skipped, and thus merged.
    """
    num_lines = len(lines)

    offset = 2
    while True:
        next_line_idx = line_idx + offset

        if next_line_idx >= num_lines:
            next_line_idx = next_line_idx % num_lines

            if (next_line_idx == line_idx or
                    next_line_idx in merged_lines or
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
    Computes the intersection of two lines defined by two points.

    Parameters
    ----------
    line_1 : (2x2) array-like
        A line defined by the coordinates of two points.
    line_2 : (2x2) array-like
        Another line defined by the coordinates of two points.

    Returns
    -------
    intersection : (1x2) tuple
        The coordinates of the intersection.

    .. [1] https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_points_on_each_line  # noqa
    """
    p1, p2 = line_1
    x1, y1 = p1
    x2, y2 = p2
    p3, p4 = line_2
    x3, y3 = p3
    x4, y4 = p4
    px = (((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) /
          ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)))
    py = (((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) /
          ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)))
    return (px, py)


def merge_offset_lines(vertices, merge_angle, max_distance):
    """
    Merges lines which have subsequent lines that are parallel and offset by a
    small distance.

    Parameters
    ----------
    vertices : (Mx2) array
        The coordinates of the vertices of the polygon.
    merge_angle : float
        The angle difference two lines may have to be considered parallel.
    max_distance : float
        The maximum distance the offset can be for the lines to be merged.

    Returns
    -------
    vertices : (Mx2) array
        The coordinates of the vertices of the resulting polygon.
    """
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
