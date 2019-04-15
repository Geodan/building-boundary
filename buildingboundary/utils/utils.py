# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import math
import numpy as np


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


def create_segments(iterable):
    """
    Creates pairs in sequence between the elements of the
    iterable.

    Parameters
    ----------
    iterable : list
        The interable to create pairs of.

    Returns
    -------
    segments : zip
        Iterator which contains pairs of points that define
        the segments.

    Examples
    --------
    >>> list(create_segments([3, 5, 1, 9, 8]))
    [(3, 5), (5, 1), (1, 9), (9, 8), (8, 3)]
    """
    return zip(iterable, np.roll(iterable, -1, axis=0))
