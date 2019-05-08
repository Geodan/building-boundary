# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import numpy as np
import pcl

from .segment import BoundarySegment


def ransac_line_segmentation(points, distance):
    """
    Fit a line using RANSAC.

    Parameters
    ----------
    points : (Mx2) array
        The coordinates of the points
    distance : float
        The maximum distance between a point and a line for a point to be
        considered belonging to that line.

    Returns
    -------
    inliers : list of int
        The indices of the inlier points
    """
    points_3 = np.vstack((points[:, 0], points[:, 1], np.zeros(len(points)))).T
    cloud = pcl.PointCloud()
    cloud.from_array(points_3.astype(np.float32))
    seg = cloud.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_LINE)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_distance_threshold(distance)
    seg.set_max_iterations(1000)
    seg.set_optimize_coefficients(False)
    inliers, _ = seg.segment()
    return inliers


def extend_segment(segment, points, indices, distance):
    line_segment = BoundarySegment(points[segment])
    line_segment.fit_line(method='TLS')

    for i in range(segment[0]-1, indices[0]-1, -1):
        if line_segment.dist_point_line(points[i]) < distance:
            segment.insert(0, i)
        else:
            break

    for i in range(segment[-1]+1, indices[-1]+1):
        if line_segment.dist_point_line(points[i]) < distance:
            segment.append(i)
        else:
            break

    return segment


def extract_segment(points, indices, distance):
    inliers = ransac_line_segmentation(points[indices], distance)
    inliers = indices[inliers]

    sequences = np.split(inliers, np.where(np.diff(inliers) != 1)[0] + 1)
    segment = list(max(sequences, key=len))

    if len(segment) > 1:
        segment = extend_segment(segment, points, indices, distance)

    return segment


def get_insert_loc(segments, segment):
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
    sequences = np.split(indices, np.where(np.diff(mask) == 1)[0] + 1)

    if mask[0]:
        sequences = [s for i, s in enumerate(sequences) if i % 2 == 0]
    else:
        sequences = [s for i, s in enumerate(sequences) if i % 2 != 0]

    sequences = [s for s in sequences if len(s) > 1]

    return sequences


def extract_segments(segments, points, indices, mask, distance):
    if len(indices) == 2:
        segment = indices
    else:
        segment = extract_segment(points, indices, distance)

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
