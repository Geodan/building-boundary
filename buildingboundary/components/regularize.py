# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import math
from itertools import combinations

from ..utils.angle import (min_angle_difference, weighted_angle_mean,
                           to_positive_angle, perpendicular)
from ..utils.error import ThresholdError
from .segmentation import merge_segments


def compute_primary_orientations(segments, num_points=float('inf'),
                                 angle_epsilon=0.1):
    max_length = -1
    orientations = []
    for i, s in enumerate(segments):
        if s.length > max_length:
            longest_segment = i
            max_length = s.length

        if len(s.points) > num_points:
            a1 = s.orientation
            for o in orientations:
                a2 = o['orientation']
                angle_diff = min_angle_difference(a1, a2)
                if angle_diff < angle_epsilon:
                    sum_length = s.length + sum(o['lengths'])
                    mean = weighted_angle_mean([to_positive_angle(a1),
                                                to_positive_angle(a2)],
                                               [s.length/sum_length,
                                                sum(o['lengths'])/sum_length])
                    o['lengths'].append(s.length)
                    o['orientation'] = mean
                    break
            else:
                orientations.append({'orientation': a1,
                                     'lengths': [s.length]})

    primary_orientations = [o['orientation'] for o in orientations]

    main_orientation = segments[longest_segment].orientation
    main_orientation90 = perpendicular(main_orientation)
    if main_orientation not in primary_orientations:
        primary_orientations.append(main_orientation)

    # if only one primary orientation is found, add an orientation
    # perpendicular to it.
    if len(primary_orientations) == 1:
        primary_orientations.append(main_orientation90)
    else:
        # add a perpendicular orientation if no approximate perpendicular
        # orientations were found
        combis = list(combinations(primary_orientations, 2))
        diffs = [min_angle_difference(c[0], c[1]) for c in combis]
        if max(diffs) < math.pi/2 - angle_epsilon:
            primary_orientations.append(main_orientation90)
        # make orientations close to perpendicular with the longest segment
        # exactly 90 degrees
        else:
            for o in orientations:
                if o['orientation'] != main_orientation:
                    ad = min_angle_difference(o['orientation'],
                                              main_orientation)
                    ad90 = abs(ad - math.pi/2)
                    if ad90 < angle_epsilon:
                        o['orientation'] = main_orientation90
            primary_orientations = list(set(o['orientation'] for
                                            o in orientations))

    return primary_orientations


def regularize_lines(boundary_segments, primary_orientations,
                     merge_angle, max_error=None):
    prev_num_segments = 0
    num_segments = len(boundary_segments)

    while num_segments != prev_num_segments:
        prev_num_segments = len(boundary_segments)
        for s in boundary_segments:
            target_orientation = s.target_orientation(primary_orientations)
            try:
                s.regularize(math.tan(target_orientation), max_error=max_error)
            except ThresholdError:
                pass

        boundary_segments, merged_segments = merge_segments(boundary_segments, merge_angle)
        num_segments = len(boundary_segments)

    for s in boundary_segments:
        target_orientation = s.target_orientation(primary_orientations)
        try:
            s.regularize(math.tan(target_orientation), max_error=max_error)
        except ThresholdError:
                pass

#    boundary_segments, merged_segments = merge_segments(boundary_segments, merge_angle)

    return boundary_segments
