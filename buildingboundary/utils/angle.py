# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import math


def min_angle_difference(a1, a2):
    """
    Returns the minimal angle difference between two orientations.

    Parameters
    ----------
    a1 : float
        An angle in radians
    a2 : float
        Another angle in radians

    Returns
    -------
    angle : float
        The minimal angle difference in radians
    """
    pos1 = abs(math.pi - abs(abs(a1 - a2) - math.pi))
    if a1 < math.pi:
        pos2 = abs(math.pi - abs(abs((a1 + math.pi) - a2) - math.pi))
    elif a2 < math.pi:
        pos2 = abs(math.pi - abs(abs(a1 - (a2 + math.pi)) - math.pi))
    else:
        return pos1
    return pos1 if pos1 < pos2 else pos2


def angle_difference(a1, a2):
    """
    Returns the angle difference between two orientations.

    Parameters
    ----------
    a1 : float
        An angle in radians
    a2 : float
        Another angle in radians

    Returns
    -------
    angle : float
        The angle difference in radians
    """
    return math.pi - abs(math.pi - abs(a1 - a2) % (math.pi*2))


def to_positive_angle(angle):
    """
    Converts an angle to positive.

    Parameters
    ----------
    angle : float
        The angle in radians

    Returns
    -------
    angle : float
        The positive angle
    """
    angle = angle % math.pi
    if angle < 0:
        angle += math.pi
    return angle


def weighted_angle_mean(angles, weights):
    """
    Takes the weighted mean of a set of angles.

    Parameters
    ----------
    angles : list
        The angles to average in radians
    weights : list
        The corrisponding weights

    Returns
    -------
    mean : float
        The weighted mean of the angles in radians
    """
    x = 0
    y = 0
    for angle, weight in zip(angles, weights):
        x += math.cos(angle) * weight
        y += math.sin(angle) * weight

    mean = math.atan2(y, x)
    return mean


def perpendicular(angle):
    """
    Returns the perpendicular angle to the given angle.

    Parameters
    ----------
    angle : float or int
        The given angle in radians

    Returns
    -------
    perpendicular : float
        The perpendicular to the given angle in radians
    """
    perp = angle + math.pi/2
    if perp > math.pi:
        perp = angle - math.pi/2
    return perp
