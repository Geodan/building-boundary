# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import math
import numpy as np


def distance(p1, p2):
    return math.hypot(*(p1-p2))


def distance_to_intersect(segment1, segment2, intersect):
    end_points = [segment1.end_points[0], segment1.end_points[1],
                  segment2.end_points[0], segment2.end_points[1]]
    distances = [distance(point, intersect) for point in end_points]
    return min(distances)


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
                slope = -(1 / segments[i + 1].slope)
            else:
                p = segments[0].end_points[0]
                slope = -(1 / segments[0].slope)

            intercept = - slope * p[0] + p[1]
            intersect = segments[i].line_intersect([slope, intercept])
            if any(intersect):
                intersections.append(intersect)
                intersections.append(p)

    return np.array(intersections)
