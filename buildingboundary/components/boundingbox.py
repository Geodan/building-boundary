# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import math
import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon


def compute_edge_angles(edges):
    """
    Compute the angles between the edges and the x-axis.

    Parameters
    ----------
    edges : (Mx2x2) array
        The coordinates of the sets of points that define the edges.

    Returns
    -------
    edge_angles : (Mx1) array
        The angles between the edges and the x-axis.
    """
    edges_count = len(edges)
    edge_angles = np.zeros(edges_count)
    for i in range(edges_count):
        edge_x = edges[i][1][0] - edges[i][0][0]
        edge_y = edges[i][1][1] - edges[i][0][1]
        edge_angles[i] = math.atan2(edge_y, edge_x)

    return np.unique(edge_angles)


def rotate_points(points, angle):
    """
    Rotate points in a coordinate system using a rotation matrix based on
    an angle.

    Parameters
    ----------
    points : (Mx2) array
        The coordinates of the points.
    angle : float
        The angle by which the points will be rotated (in radians).

    Returns
    -------
    points_rotated : (Mx2) array
        The coordinates of the rotated points.
    """
    # Compute rotation matrix
    rot_matrix = np.array(((math.cos(angle), -math.sin(angle)),
                           (math.sin(angle), math.cos(angle))))
    # Apply rotation matrix to the points
    points_rotated = np.dot(points, rot_matrix)

    return np.array(points_rotated)


def rotating_calipers_bbox(points, angles):
    """
    Compute the oriented minimum bounding box using a rotating calipers
    algorithm.

    Parameters
    ----------
    points : (Mx2) array
        The coordinates of the points of the convex hull.
    angles : (Mx1) array-like
        The angles the edges of the convex hull and the x-axis.

    Returns
    -------
    corner_points : (4x2) array
        The coordinates of the corner points of the minimum oriented
        bounding box.
    """
    min_bbox = {'angle': 0,
                'minmax': (0, 0, 0, 0),
                'area': float('inf')}

    for a in angles:
        # Rotate the points and compute the new bounding box
        rotated_points = rotate_points(points, a)
        min_x = min(rotated_points[:, 0])
        max_x = max(rotated_points[:, 0])
        min_y = min(rotated_points[:, 1])
        max_y = max(rotated_points[:, 1])
        area = (max_x - min_x) * (max_y - min_y)

        # Save if the new bounding box is smaller than the current smallest
        if area < min_bbox['area']:
            min_bbox = {'angle': a,
                        'minmax': (min_x, max_x, min_y, max_y),
                        'area': area}

    # Extract the rotated corner points of the minimum bounding box
    c1 = (min_bbox['minmax'][0], min_bbox['minmax'][2])
    c2 = (min_bbox['minmax'][0], min_bbox['minmax'][3])
    c3 = (min_bbox['minmax'][1], min_bbox['minmax'][3])
    c4 = (min_bbox['minmax'][1], min_bbox['minmax'][2])
    rotated_corner_points = [c1, c2, c3, c4]

    # Rotate the corner points back to the original system
    corner_points = np.array(rotate_points(rotated_corner_points,
                                           2*math.pi-min_bbox['angle']))

    return corner_points


def compute_bounding_box(points, given_angles=None):
    """
    Computes the minimum area oriented bounding box of a set of points.

    Parameters
    ----------
    points : (Mx2) array
        The coordinates of the points.

    Returns
    -------
    bbox : polygon
        The minimum area oriented bounding box as a shapely polygon.
    """
    hull = ConvexHull(points).simplices
    hull_unique_i = np.array(list(set([p for s in hull for p in s])))
    hull_points = points[hull_unique_i]
    if given_angles is None:
        angles = compute_edge_angles(points[hull])
    else:
        angles = given_angles
    bbox_corner_points = rotating_calipers_bbox(hull_points, angles)
    bbox = Polygon(bbox_corner_points)
    return bbox
