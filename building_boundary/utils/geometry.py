# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import math


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


def perpedicular_line(line, p):
    """
    Returns a perpendicular line to a line at a point.

    Parameters
    ----------
    line : (1x3) array-like
        The a, b, and c coefficients (ax + by + c = 0) of a line.
    p : (1x2) array-like
        The coordinates of a point on the line.

    Returns
    -------
    line : (1x3) array-like
        The a, b, and c coefficients (ax + by + c = 0) of the line
        perpendicular to the input line at point p.
    """
    a, b, c = line
    pa = b
    pb = -a
    pc = -(p[0] * b - p[1] * a)
    return [pa, pb, pc]
