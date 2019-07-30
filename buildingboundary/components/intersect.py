# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import numpy as np

from ..utils import distance


def perpedicular_line(line, p):
    """
    Returns a perpendicular line to a line at a point.

    Parameters
    ----------
    line : (1x3) array-like
        The a, b, and c coefficients (ax + by + c = 0) of a line.
    p : (1x2) array-like
        The coordinates of a point on the line.

    Returns
    -------
    line : (1x3) array-like
        The a, b, and c coefficients (ax + by + c = 0) of the line
        perpendicular to the input line at point p.
    """
    a, b, c = line
    pa = b
    pb = -a
    pc = -(p[0] * b - p[1] * a)
    return [pa, pb, pc]


def perpedicular_line_intersect(segment1, segment2):
    perp_line = perpedicular_line(segment1.line,
                                  segment1.end_points[1])
    return segment2.line_intersect(perp_line)


def intersect_distance(intersect, segment1, segment2):
    return min(distance(segment1.end_points[1], intersect),
               distance(segment2.end_points[0], intersect))


def compute_intersections(segments, perp_dist_weight=3):
    """
    Computes the intersections between the segments in sequence. If
    no intersection could be found or the perpendicular line results in
    an intersection closer to the segments a perpendicular line will be
    added in between the two segments.

    Parameters
    ----------
    segments : list of BoundarySegment
        The wall segments to compute intersections for.

    Returns
    -------
    intersections : (Mx1) array
        The computed intersections.
    """
    intersections = []
    num_segments = len(segments)
    for i in range(num_segments):
        segment1 = segments[i]
        segment2 = segments[(i + 1) % num_segments]

        intersect = segment1.line_intersect(segment2.line)

        if any(intersect):
            intersect_dist = intersect_distance(intersect, segment1, segment2)
            perp_intersect = perpedicular_line_intersect(segment1, segment2)
            if any(perp_intersect):
                perp_intersect_dist = intersect_distance(perp_intersect,
                                                         segment1,
                                                         segment2)

                if intersect_dist > perp_intersect_dist * perp_dist_weight:
                    intersections.append(segment1.end_points[1])
                    intersections.append(perp_intersect)
                else:
                    intersections.append(intersect)
            else:
                intersections.append(intersect)
        else:
            # if no intersection was found add a perpendicular line at the end
            # and intersect using the new line
            intersect = perpedicular_line_intersect(segment1, segment2)
            intersections.append(segment1.end_points[1])
            intersections.append(intersect)

    return np.array(intersections)
