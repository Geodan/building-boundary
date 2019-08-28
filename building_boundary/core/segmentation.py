# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import numpy as np
from skimage.measure import LineModelND, ransac
from .segment import BoundarySegment


def ransac_line_segmentation(points, distance):
    """
    Segment a line using RANSAC.

    Parameters
    ----------
    points : (Mx2) array
        The coordinates of the points
    distance : float
        The maximum distance between a point and a line for a point to be
        considered belonging to that line.

    Returns
    -------
    inliers : list of bool
        True where point is an inlier.
    """
    _, inliers = ransac(points, LineModelND,
                        min_samples=2,
                        residual_threshold=distance,
                        max_trials=1000)
    return inliers


def extend_segment(segment, points, indices, distance):
    """
    Extend a line found by ransac based on the sequence and the distance.

    Parameters
    ----------
    segment : list of int
        The indices of the points belonging to the segment/line.
    points : (Mx2) array
        The coordinates of all the points.
    indices : list of int
        The indices of the points in the sequence.
    distance : float
        The maximum distance between a point and a line for a point to be
        considered belonging to that line.

    Returns
    -------
    segment : list of int
        The indices of the points belonging to the segment/line.
    """
    line_segment = BoundarySegment(points[segment])

    for i in range(segment[0]-1, indices[0]-1, -1):
        if line_segment.dist_point_line(points[i]) < distance:
            segment.insert(0, i)
        else:
            if (i - 2 >= indices[0] and
                    line_segment.dist_point_line(points[i-1]) < distance and
                    line_segment.dist_point_line(points[i-2]) < distance):
                continue
            else:
                break

    for i in range(segment[-1]+1, indices[-1]+1):
        if line_segment.dist_point_line(points[i]) < distance:
            segment.append(i)
        else:
            if (i + 2 <= indices[-1] and
                    line_segment.dist_point_line(points[i+1]) < distance and
                    line_segment.dist_point_line(points[i+2]) < distance):
                continue
            else:
                break

    return segment


def extract_segment(points, indices, distance):
    """
    Extract a line segment from a sequence of points.

    Parameters
    ----------
    points : (Mx2) array
        The coordinates of all the points.
    indices : list of int
        The indices of the points in the sequence.
    distance : float
        The maximum distance between a point and a line for a point to be
        considered belonging to that line.

    Returns
    -------
    segment : list of int
        The indices of the points belonging to the segment/line.
    """
    inliers = ransac_line_segmentation(points[indices], distance)
    inliers = indices[inliers]

    sequences = np.split(inliers, np.where(np.diff(inliers) != 1)[0] + 1)
    segment = list(max(sequences, key=len))

    if len(segment) > 1:
        segment = extend_segment(segment, points, indices, distance)
    elif len(segment) == 1:
        if segment[0] + 1 in indices:
            segment.append(segment[0] + 1)
            segment = extend_segment(segment, points, indices, distance)
        elif segment[0] - 1 in indices:
            segment.insert(0, segment[0] - 1)
            segment = extend_segment(segment, points, indices, distance)

    return segment


def get_insert_loc(segments, segment):
    """
    Uses a binary search to find the correct location to insert a new segment.

    Parameters
    ----------
    segments : list of list of int
        The indices of the points belonging to the segments/lines.
    segment : list of int
        The indices of the points belonging to the segment/line.

    Returns
    -------
     : int
        The index where the segment should be inserted.
    """
    if len(segments) == 0:
        return 0
    if segment[0] > segments[-1][0]:
        return len(segments)

    lo = 0
    hi = len(segments)
    while lo < hi:
        mid = (lo + hi) // 2
        if segment[0] < segments[mid][0]:
            hi = mid
        else:
            lo = mid + 1
    return lo


def get_remaining_sequences(indices, mask):
    """
    Gets the remaining sequences given the points that are already part of
    a segment.

    Parameters
    ----------
    indices : list of int
        The indices of the points in the sequence.
    mask : list of bool
        Marks the points that are part of a segment.

    Returns
    -------
    sequences : list of list of int
        The indices of each remaining sequence.
    """
    sequences = np.split(indices, np.where(np.diff(mask) == 1)[0] + 1)

    if mask[0]:
        sequences = [s for i, s in enumerate(sequences) if i % 2 == 0]
    else:
        sequences = [s for i, s in enumerate(sequences) if i % 2 != 0]

    sequences = [s for s in sequences if len(s) > 1]

    return sequences


def extract_segments(segments, points, indices, mask, distance):
    """
    Extract line segments from a ring of points.

    Note: This is a recurisve function. Initiate with an empty segments
          list.

    Parameters
    ----------
    segments : list of list of int
        The indices of the points belonging to segments/lines.
    points : (Mx2) array
        The coordinates of all the points.
    indices : list of int
        The indices of the points in the sequence.
    mask : list of bool
        Marks the points that are part of a segment.
    distance : float
        The maximum distance between a point and a line for a point to be
        considered belonging to that line.
    """
    if len(indices) == 2:
        segment = indices
    else:
        segment = extract_segment(points, indices, distance)

    if len(segment) > 1:
        insert_loc = get_insert_loc(segments, segment)
        segments.insert(insert_loc, segment)

        mask[segment[0]:segment[-1]+1] = False

        sequences = get_remaining_sequences(indices, mask[indices])

        for s in sequences:
            extract_segments(segments, points, s, mask, distance)


def boundary_segmentation(points, distance):
    """
    Extract linear segments using RANSAC.

    Parameters
    ----------
    points : (Mx2) array
        The coordinates of the points.
    distance : float
        The maximum distance between a point and a line for a point to be
        considered belonging to that line.

    Returns
    -------
    segments : list of array
        The linear segments.
    """
    points_shifted = points.copy()
    shift = np.min(points_shifted, axis=0)
    points_shifted -= shift

    mask = np.ones(len(points_shifted), dtype=np.bool)
    indices = np.arange(len(points_shifted))

    segments = []
    extract_segments(segments, points_shifted, indices, mask, distance)

    segments = [points_shifted[i]+shift for i in segments]

    return segments
