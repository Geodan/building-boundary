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


def min_distance_segments_point(segment1, segment2, point):
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


def compute_intersections(segments, max_distance=float('inf')):
    """
    Computes the intersections between the segments in sequence. If
    no intersection could be found or the intersection is at a further
    distance than the given max distance, a perpendicular line will be
    added in between the two segments.

    Parameters
    ----------
    segments : list of BoundarySegment
        The wall segments to compute intersections for.
    max_distance : float or int, optional
        The maximum distance between an intersection and the
        corrisponding segments.

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

        intersect = segment1.line_intersect([segment2.slope,
                                             segment2.intercept])

        if any(intersect):
            intersect_distance = min_distance_segments_point(segment1.end_points,
                                                             segment2.end_points,
                                                             intersect)
            if intersect_distance < max_distance:
                intersections.append(intersect)
            else:
                intersect = np.array([])

        if not any(intersect):
            # if no intersection was found add a perpendicular line at the end
            # and intersect using the new line
            if i != num_segments - 1:
                p = segments[i + 1].end_points[0]
                slope = (-(1 / segments[i + 1].slope)
                         if segments[i + 1].slope != 0
                         else -(1 / 0.00000001))
            else:
                p = segments[0].end_points[0]
                slope = (-(1 / segments[0].slope)
                         if segments[0].slope != 0
                         else -(1 / 0.00000001))

            intercept = - slope * p[0] + p[1]
            intersect = segments[i].line_intersect([slope, intercept])
            if any(intersect):
                intersections.append(intersect)
                intersections.append(p)

    return np.array(intersections)
