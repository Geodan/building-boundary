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
        if i != num_segments - 1:
            segment2 = segments[i + 1]
        else:
            segment2 = segments[0]

        intersect = segment1.line_intersect([segment2.a,
                                             segment2.b,
                                             segment2.c])

        if any(intersect):
            intersect_dist = min(distance(segment1.end_points[1], intersect),
                                 distance(segment2.end_points[0], intersect))

            line = perpedicular_line([segment1.a, segment1.b, segment1.c],
                                     segment1.end_points[1])
            perp_intersect = segment2.line_intersect(line)
            if any(perp_intersect):
                perp_intersect_dist = min(distance(segment1.end_points[1],
                                                   perp_intersect),
                                          distance(segment2.end_points[0],
                                                   perp_intersect))

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
            line = perpedicular_line([segment1.a, segment1.b, segment1.c],
                                     segment1.end_points[1])
            intersect = segment2.line_intersect(line)
            intersections.append(segment1.end_points[1])
            intersections.append(intersect)

    return np.array(intersections)
