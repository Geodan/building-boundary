# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import math
import numpy as np

from ..utils.error import ThresholdError
from ..utils.angle import min_angle_difference


def PCA(points):
    """
    Does a Principle Component Analysis (PCA) for a set of 3D
    points (the structure tensor) by computing the eigenvalues
    and eigenvectors of the covariance matrix of a point cloud.

    Parameters
    ----------
    points : (Mx3) array
        X, Y and Z coordinates of points.

    Returns
    -------
    eigenvalues : (1x3) array
        The eigenvalues corrisponding to the eigenvectors of the covariance
        matrix.
    eigenvectors : (3x3) array
        The eigenvectors of the covariance matrix.
    """
    cov_mat = np.cov(points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov_mat)
    order = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    return eigenvalues, eigenvectors


class BoundarySegment(object):
    def __init__(self, points):
        """
        Initiate a boundary segment for a set of points.

        Parameters
        ----------
        points : (Mx2) array
            X and Y coordinates of points.

        Attributes
        ----------
        points : (Mx2) array
            X and Y coordinates of points.
        """
        self.points = points

    def slope(self):
        return -self.a / self.b

    def intercept(self):
        return -self.c / self.b

    def fit_line(self, method='TLS', max_error=None):
        """
        Fit a line to the set of points of the object.

        Parameters
        ----------
        method : string
            The method used to fit the line. Options:
            - Ordinary Least Squares: 'OLS'
            - Total Least Squares: 'TLS'
        max_error : float or int
            The maximum error (average distance points to line) the
            fitted line is allowed to have. A ThresholdError will be
            raised if this max error is exceeded.

        Attributes
        ----------
        slope : float
            The slope of the fitted line.
        intercept : float
            The y-intercept of the fitted line.

        Raises
        ------
        NotImplementedError
            If a non-existing method is chosen.
        ThresholdError
            If the error of the fitted line (average distance points to
            line) exceeds the given max error.
        """
        if len(self.points) == 1:
            raise ValueError('Not enough points to fit a line.')
        elif len(self.points) == 2:
            dx, dy = np.diff(self.points, axis=0)[0]
            if dx == 0:
                self.a = 0
            else:
                self.a = dy / dx
            self.b = -1
            self.c = (np.mean(self.points[:, 1]) -
                      np.mean(self.points[:, 0]) * self.a)
        elif all(self.points[0, 0] == self.points[:, 0]):
            self.a = 1
            self.b = 0
            self.c = -self.points[0, 0]
        elif all(self.points[0, 1] == self.points[:, 1]):
            self.a = 0
            self.b = 1
            self.c = -self.points[0, 1]
        else:
            if method == 'OLS':
                self.a, self.c = np.polyfit(self.points[:, 0],
                                            self.points[:, 1], 1)
                self.b = -1
            elif method == 'TLS':
                _, eigenvectors = PCA(self.points)
                self.a = eigenvectors[1, 0] / eigenvectors[0, 0]
                self.b = -1
                self.c = (np.mean(self.points[:, 1]) -
                          np.mean(self.points[:, 0]) * self.a)
            else:
                raise NotImplementedError("Chosen method not available.")

            if max_error is not None:
                residuals = self.residuals()
                if residuals > max_error:
                    raise ThresholdError("Could not fit a proper line. "
                                         "Error: {}".format(residuals))

        self._create_line_segment()

    def _point_on_line(self, point):
        """
        Finds the closest point on the fitted line from another point.

        Parameters
        ----------
        point : (1x2) array
            The X and Y coordinates of a point.

        Returns
        -------
        point : (1x2) array
            The X and Y coordinates of the closest point to the given
            point on the fitted line.

        .. [1] https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Line_defined_by_an_equation
        """
        if self.a == 0 and self.b == 0:
            raise ValueError('Invalid line. Line coefficients a and b '
                             '(ax + by + c = 0) cannot both be zero.')

        x = (self.b * (self.b * point[0] - self.a * point[1]) -
             self.a * self.c) / (self.a**2 + self.b**2)
        y = (self.a * (-self.b * point[0] + self.a * point[1]) -
             self.b * self.c) / (self.a**2 + self.b**2)
        return [x, y]

    def _create_line_segment(self):
        """
        Defines a line segment of the fitted line by creating
        the end points, length and orientation.

        Attributes
        ----------
        end_points : (2x2) array
            The coordinates of the end points of the line segment.
        length : float
            The length of the line segment.
        orientation : float
            The orientation of the line segment (in radians).

        Raises
        ------
        ValueError
            If not enough points exist to create a line segment.
        """
        if len(self.points) == 1:
            raise ValueError('Not enough points to create a line.')
        else:
            start_point = self._point_on_line(self.points[0])
            end_point = self._point_on_line(self.points[-1])
            self.end_points = np.array([start_point, end_point])
            dx, dy = np.diff(self.end_points, axis=0)[0]
            self.length = math.hypot(dx, dy)
            self.orientation = math.atan2(dy, dx)

    def residuals(self):
        """
        Computes the max distance between the points and the fitted
        line.

        Returns
        -------
        error : float
            The max distance between the points and the fitted line.
        """
        self.dist_points_line()

        return max(abs(self.distances))

    def dist_points_line(self):
        """
        Computes the distances from each point to the fitted line.

        Attributes
        ----------
        distances : (1xN) array
            The distances from each point to the fitted line.

        .. [1] https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
        """
        self.distances = (abs(self.a * self.points[:, 0] +
                              self.b * self.points[:, 1] + self.c) /
                          math.sqrt(self.a**2 + self.b**2))

    def dist_point_line(self, point):
        """
        Computes the distance from the given point to the fitted line.

        Parameters
        ----------

        Returns
        -------

        .. [1] https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
        """
        distance = (abs(self.a * point[0] + self.b * point[1] + self.c) /
                    math.sqrt(self.a**2 + self.b**2))
        return distance

    def side_points_line(self):
        """
        Determines on which side the points lie from the line segment.

        Attributes
        ----------
        sides : (1xN) array
            The side each point lies from the line segment. `0` is on the line,
            `1` is on one side, `-1` on the other.

        .. [1] https://stackoverflow.com/questions/1560492/how-to-tell-whether-a-point-is-to-the-right-or-left-side-of-a-line
        """
        sides = ((self.end_points[1, 0] - self.end_points[0, 0]) *
                 (self.points[:, 1] - self.end_points[0, 1]) -
                 (self.end_points[1, 1] - self.end_points[0, 1]) *
                 (self.points[:, 0] - self.end_points[0, 0]))
        self.sides = np.sign(sides)

    def inflate(self, order='cw'):
        """
        Moves the line segments to the outer most point of the object.
        """
        self.dist_points_line()
        self.side_points_line()

        if order == 'cw':
            outside_points = np.where(self.sides == 1)[0]
        elif order == 'ccw':
            outside_points = np.where(self.sides == -1)[0]
        else:
            raise ValueError('Invalid value for order.'
                             'Must be either \'cw\' or \'ccw\'.')

        if len(outside_points) == 0:
            return

        furthest_outside_point_idx = np.argmax(self.distances[outside_points])
        furthest_point_idx = outside_points[furthest_outside_point_idx]
        furthest_point = self.points[furthest_point_idx]
        point_on_line = self._point_on_line(furthest_point)
        dx, dy = furthest_point - point_on_line
        self.c = self.c - self.a * dx - self.b * dy
        self._create_line_segment()

    def target_orientation(self, primary_orientations):
        """
        Determines the which of the given primary orientations is closest
        to the orientation of this line segment.

        Parameters
        ----------
        primary_orientations : list of float
            The determined primary orientations.

        Returns
        -------
        The primary orientation closest to the orientation of this line
        segment.
        """
        po_diff = [min_angle_difference(self.orientation, o) for
                   o in primary_orientations]
        min_po_diff = min(po_diff)
        return primary_orientations[po_diff.index(min_po_diff)]

    def regularize(self, slope, max_error=None):
        """
        Recreates the line segment based on the given slope.

        Parameters
        ----------
        slope : float or int
            The slope the line segment should have.
        max_error : float or int
            The maximum error (average distance points to line) the
            fitted line is allowed to have. A ThresholdError will be
            raised if this max error is exceeded.

        Raises
        ------
        ThresholdError
            If the error of the fitted line (average distance points to
            line) exceeds the given max error.

        .. [1] https://math.stackexchange.com/questions/1377716/how-to-find-a-least-squares-line-with-a-known-slope
        """
        if not np.isclose(slope, math.tan(self.orientation)):
            self.a = slope
            self.b = -1
            self.c = (sum(self.points[:, 1] - self.a * self.points[:, 0]) /
                      len(self.points))

            if max_error is not None:
                residuals = self.residuals()
                if residuals > max_error:
                    raise ThresholdError("Could not fit a proper line. "
                                         "Error: {}".format(residuals))

            self._create_line_segment()

    def line_intersect(self, line):
        """
        Compute the intersection between this line and another.

        Parameters
        ----------
        line : (1x3) array-like
            The a,b, and c coefficients (ax + by + c = 0) of a line.

        Returns
        -------
        point : (1x2) array
            The coordinates of intersection. Returns empty array if no
            intersection found.
        """
        a, b, c = line
        d = self.a * b - self.b * a
        if d != 0:
            dx = -self.c * b + self.b * c
            dy = self.c * a - self.a * c
            x = dx / float(d)
            y = dy / float(d)
            return np.array([x, y])
        else:
            return np.array([])
