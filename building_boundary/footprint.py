# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import math

import numpy as np
from shapely.geometry import (
    Polygon, MultiPolygon, LineString, MultiLineString, LinearRing
)
from shapely import wkt

from building_boundary import utils


def line_orientations(lines):
    """
    Computes the orientations of the lines.

    Parameters
    ----------
    lines : list of (2x2) array
        The lines defined by the coordinates two points.

    Returns
    -------
    orientations : list of float
        The orientations of the lines in radians from
        0 to pi (east to west counterclockwise)
        0 to -pi (east to west clockwise)
    """
    orientations = []
    for l in lines:
        dx, dy = l[0] - l[1]
        orientation = math.atan2(dy, dx)
        if not any([np.isclose(orientation, o) for o in orientations]):
            orientations.append(orientation)
    return orientations


def geometry_orientations(geom):
    """
    Computes the orientations of the lines of a geometry (Polygon,
    MultiPolygon, LineString, MultiLineString, or LinearRing).

    Parameters
    ----------
    geom : Polygon, MultiPolygon, LineString, MultiLineString, or LinearRing
        The geometry

    Returns
    -------
    orientations : list of float
        The orientations of the lines of the geometry in radians from
        0 to pi (east to west counterclockwise)
        0 to -pi (east to west clockwise)
    """
    orientations = []
    if type(geom) == Polygon:
        lines = utils.create_pairs(geom.exterior.coords[:-1])
        orientations = line_orientations(lines)
    elif type(geom) == MultiPolygon:
        for p in geom:
            lines = utils.create_pairs(p.exterior.coords[:-1])
            orientations.extend(line_orientations(lines))
    elif type(geom) == LineString:
        if geom.coords[0] == geom.coords[-1]:
            lines = utils.create_pairs(geom.coords[:-1])
        else:
            lines = list(utils.create_pairs(geom.coords))[:-1]
        orientations = line_orientations(lines)
    elif type(geom) == MultiLineString:
        for l in geom:
            if l.coords[0] == l.coords[-1]:
                lines = utils.create_pairs(l.coords[:-1])
            else:
                lines = list(utils.create_pairs(l.coords))[:-1]
            orientations.extend(line_orientations(lines))
    elif type(geom) == LinearRing:
        lines = utils.create_pairs(geom.coords[:-1])
        orientations = line_orientations(lines)
    else:
        raise TypeError('Invalid geometry type. Expects Polygon, '
                        'MultiPolygon, LineString, MultiLineString, '
                        'or LinearRing.')
    return orientations


def compute_orientations(footprint_wkt):
    """
    Computes the orientations of the footprint.

    Parameters
    ----------
    footprint_wkt : string
        The footprint geometry defined by a WKT string.

    Returns
    -------
    orientations : list of float
        The orientations of the lines of the geometry in radians from
        0 to pi (east to west counterclockwise)
        0 to -pi (east to west clockwise)
    """
    footprint_geom = wkt.loads(footprint_wkt)
    orientations = geometry_orientations(footprint_geom)
    return orientations
