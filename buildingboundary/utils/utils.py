# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import numpy as np


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
    return zip(iterable, np.roll(iterable, -1))
