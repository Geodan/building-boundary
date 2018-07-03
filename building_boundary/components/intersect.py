# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import numpy as np


def compute_intersections(segments):
    intersections = []
    num_segments = len(segments)
    for i in range(num_segments):
        if i != num_segments - 1:
            intersect = segments[i].line_intersect([segments[i + 1].slope,
                                                    segments[i + 1].intersept])
        else:
            intersect = segments[i].line_intersect([segments[0].slope,
                                                    segments[0].intersept])

        if intersect:
            intersections.append(intersect)
        else:
            # if no intersection was found add a perpendicular line at the end
            # and intersect using the new line
            if i != num_segments - 1:
                p = segments[i + 1].end_points[0]
                slope = -(1 / segments[i + 1].slope)
            else:
                p = segments[0].end_points[0]
                slope = -(1 / segments[0].slope)

            intersept = - slope * p[0] + p[1]
            intersect = segments[i].line_intersect([slope, intersept])
            if intersect:
                intersections.append(intersect)
                intersections.append(p)

    return np.array(intersections)
