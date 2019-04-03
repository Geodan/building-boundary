# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import numpy as np
import pcl
from scipy.spatial import ConvexHull

from ..utils.error import ThresholdError
from ..utils.utils import create_segments
from .segment import BoundarySegment


def convex_fit(points, boundary_segments, max_error):
    """
    Recursive function which segments a set of points into
    linear segments.

    Parameters
    ----------
    points : (Mx2) array
        The coordinates of the points
    boundary_segments : list of BoundarySegment
        The accepted boundary segments. Initiate using a variable
        containing an empty list.
    max_error : float or int
        The maximum error (average distance points to line) the
        fitted line is allowed to have.
    """
    hull = ConvexHull(points)
    hull.vertices.sort()
    segments = create_segments(hull.vertices)

    for s in segments:
        if s[1]+1 == len(hull.points) or s[1] == 0:
            s_points = hull.points[s[0]:]
        else:
            s_points = hull.points[s[0]:s[1]+1]

        if len(s_points) == 1:
            continue

        try:
            segment = BoundarySegment(s_points)
            segment.fit_line(method='TLS', max_error=max_error)
            boundary_segments.append(segment)
        except ThresholdError:
            convex_fit(s_points, boundary_segments, max_error=max_error)


def ransac_line_segmentation(points, distance, iterations=1000):
    """
    Fit a line using RANSAC.

    Parameters
    ----------
    points : (Mx2) array
        The coordinates of the points
    distance : float
        The maximum distance between a point and a line for a point to be
        considered belonging to that line.
    iterations : int, optional
        The number of iterations within the RANSAC algorithm.

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
    seg.set_max_iterations(iterations)
    seg.set_optimize_coefficients(False)
    inliers, _ = seg.segment()
    return inliers


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


def extract_segments_ransac(points, min_size=2, distance=0.3, iterations=1000):
    """
    Extract linear segments using RANSAC.

    Parameters
    ----------
    points : (Mx2) array
        The coordinates of the points.
    min_size : int
        The minimum size a segment needs to be to be considered a relevant
        segment/line.
    distance : float
        The maximum distance between a point and a line for a point to be
        considered belonging to that line.
    iterations : int
        The number of iterations within the RANSAC algorithm.

    Returns
    -------
    segments : list of array
        The linear segments.
    """
    shift = np.min(points, axis=0)
    points -= shift

    segments = []
    mask = np.ones(len(points), dtype=np.bool)
    indices = np.arange(len(points))

    while True:
        current_points = points[mask]
        assert len(indices) == len(current_points)

        inliers = ransac_line_segmentation(current_points, distance)
        inliers = indices[inliers]

        sequences = np.split(inliers, np.where(np.diff(inliers) != 1)[0] + 1)
        longest_seq = max(sequences, key=len)

        segment = list(range(longest_seq[0], longest_seq[-1]+1))

        if len(segment) < min_size:
            break

        insert_loc = get_insert_loc(segments, segment)
        segments.insert(insert_loc, segment)

        mask[longest_seq[0]:longest_seq[-1]+1] = False

        if len(points[mask]) < min_size:
            break

        if np.all(mask) is False:
            break

        indices = np.array([i for i in indices if i not in longest_seq])

    segments = [points[i] for i in segments]

    return segments
