# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import numpy as np

from .. import utils


def perpedicular_line_intersect(segment1, segment2):
    perp_line = utils.geometry.perpedicular_line(segment1.line,
                                                 segment1.end_points[1])
    return segment2.line_intersect(perp_line)


def intersect_distance(intersect, segment1, segment2):
    return min(utils.geometry.distance(segment1.end_points[1], intersect),
               utils.geometry.distance(segment2.end_points[0], intersect))


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
                if ((intersect_dist >
                        perp_intersect_dist * perp_dist_weight) or
                        (segment2.side_point_on_line(intersect) == -1 and
                         perp_intersect_dist <
                         intersect_dist * perp_dist_weight)):
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
