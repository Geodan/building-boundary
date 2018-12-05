# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import math
import numpy as np


def distance(p1, p2):
    return math.hypot(*(p1-p2))


def dist_point_to_line_segment(point, segment):
    # http://geomalgorithms.com/a02-_lines.html
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


def distance_to_intersect(segment1, segment2, intersect):
    distance1 = dist_point_to_line_segment(intersect, segment1.end_points)
    distance2 = dist_point_to_line_segment(intersect, segment2.end_points)
    return min([distance1, distance2])


def compute_intersections(segments, max_interect_distance=float('inf')):
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
            intersect_distance = distance_to_intersect(segment1, segment2,
                                                       intersect)
            if intersect_distance < max_interect_distance:
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
