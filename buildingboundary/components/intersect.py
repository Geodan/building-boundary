# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import math
import numpy as np


def distance(p1, p2):
    """
    The euclidean distance between two points.

    Parameters
    ----------
    p1 : list or array
        A point in 2D space.
    p2 : list or array
        A point in 2D space.

    Returns
    -------
    distance : float
        The euclidean distance between the two points.
    """
    return math.hypot(*(p1-p2))


def distance_point_segment(point, segment):
    """
    The euclidean distance between between a point and
    a line segment.

    ref: http://geomalgorithms.com/a02-_lines.html

    Parameters
    ----------
    point : list or array
        A point in 2D space.
    segment : list or array
        A line segment defined by two points.

    Returns
    -------
    distance : float
        The euclidean distance between the point and
        the line segment.
    """
    #
    v = segment[1] - segment[0]
    w = point - segment[0]

    c1 = np.dot(w, v)
    if c1 <= 0:
        return distance(point, segment[0])

    c2 = np.dot(v, v)
    if c2 <= c1:
        return distance(point, segment[1])

    b = c1 / c2
    pb = segment[0] + b * v
    return distance(point, pb)


def min_dist_segments_point(segment1, segment2, point):
    """
    The minimum euclidean distance between a point and
    two line segments.

    Parameters
    ----------
    segment1 : list or array
        A line segment defined by two points.
    segment2 : list or array
        A line segment defined by two points.
    point : list or array
        A point in 2D space.

    Returns
    -------
    distance : float
        The minimum euclidean distance between the point and
        the two line segments.
    """
    distance1 = distance_point_segment(point, segment1)
    distance2 = distance_point_segment(point, segment2)
    return min([distance1, distance2])


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


def compute_intersections(segments):
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
            intersect_dist = min_dist_segments_point(segment1.end_points,
                                                     segment2.end_points,
                                                     intersect)

            line = perpedicular_line([segment1.a, segment1.b, segment1.c],
                                     segment1.end_points[1])
            perp_intersect = segment2.line_intersect(line)
            perp_intersect_dist = min_dist_segments_point(segment1.end_points,
                                                          segment2.end_points,
                                                          perp_intersect)

            if intersect_dist > perp_intersect_dist:
                intersections.append(segment1.end_points[1])
                intersections.append(perp_intersect)
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
