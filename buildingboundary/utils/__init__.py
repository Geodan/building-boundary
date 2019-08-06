# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import numpy as np

from . import angle
from . import geometry
from . import error


def create_pairs(iterable):
    """
    Creates pairs in sequence between the elements of the
    iterable.

    Parameters
    ----------
    iterable : list
        The interable to create pairs of.

    Returns
    -------
    pairs : zip
        Iterator which contain the pairs.

    Examples
    --------
    >>> list(create_pairs([3, 5, 1, 9, 8]))
    [(3, 5), (5, 1), (1, 9), (9, 8), (8, 3)]
    """
    return zip(np.array(iterable), np.roll(iterable, -1, axis=0))


__all__ = [
    'angle',
    'geometry',
    'error',
    'orientations',
    'create_pairs'
]
