# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import math
import numpy as np

from ..utils.error import ThresholdError
from ..utils.angle import min_angle_difference


def PCA(points):
    cov_mat = np.cov(points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov_mat)
    order = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    return eigenvalues, eigenvectors


class BoundarySegment(object):
    def __init__(self, points):
        self.points = points

    def fit_line(self, method='OLS', max_error=None):
        if len(self.points) == 1:
            raise ValueError('Not enough points to fit a line.')
        elif len(self.points) == 2:
            dx, dy = np.diff(self.points, axis=0)[0]
            if dx == 0:
                dx = 0.000001
            self.slope = dy / dx
            self.intercept = (np.mean(self.points[:, 1]) -
                              np.mean(self.points[:, 0]) * self.slope)
        else:
            if method == 'OLS':
                self.slope, self.intercept = np.polyfit(self.points[:, 0],
                                                        self.points[:, 1], 1)
            elif method == 'TLS':
                _, eigenvectors = PCA(self.points)
                self.slope = eigenvectors[1, 0] / eigenvectors[0, 0]
                self.intercept = (np.mean(self.points[:, 1]) -
                                  np.mean(self.points[:, 0]) * self.slope)
            else:
                raise NotImplementedError("Chosen method not available.")

            if max_error is not None:
                residuals = self.residuals()
                if residuals > max_error:
                    raise ThresholdError("Could not fit a proper line. Error:\
                                         {}".format(residuals))

        self._create_line()

    @staticmethod
    def _distance(p1, p2):
        return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

    def _create_line(self):
        def f(x): return self.slope * x + self.intersept
        x = np.array([self.points[0, 0], self.points[-1, 0]])
        y = f(x)
        self.end_points = np.array((x, y)).T

        if (self._distance(self.end_points[0], self.points[0]) >
                self._distance(self.end_points[0], self.points[-1])):
            self.end_points = self.end_points[::-1]

        dx = np.diff(x)
        dy = np.diff(y)
        self.length = math.hypot(abs(dx), abs(self.points[0, 1] -
                                              self.points[-1, 1]))
#        self.length = math.hypot(dx, dy)
        self.orientation = math.atan2(dy, dx)

    def residuals(self):
        self.dist_points_line()

        return sum(abs(self.distances)) / len(self.points)

    def dist_points_line(self):
        # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
        self.distances = (abs(self.slope * self.points[:, 0] +
                              -self.points[:, 1] + self.intercept) /
                          math.sqrt(self.slope ** 2 + 1))

    def side_points_line(self):
        # https://stackoverflow.com/questions/1560492/how-to-tell-whether-a-point-is-to-the-right-or-left-side-of-a-line
        sides = ((self.end_points[1, 0] - self.end_points[0, 0]) *
                 (self.points[:, 1] - self.end_points[0, 1]) -
                 (self.end_points[1, 1] - self.end_points[0, 1]) *
                 (self.points[:, 0] - self.end_points[0, 0]))
        self.sides = np.sign(sides)

    def inflate(self):
        self.dist_points_line()
        self.side_points_line()

        outside_points = np.where(self.sides == -1)[0]
        if len(outside_points) == 0:
            return

        furthest_point = np.argmax(self.distances[outside_points])
        furthest_point = outside_points[furthest_point]
        d = self.distances[furthest_point]

        if self.orientation < math.pi/2 and self.orientation > 0:
            angle = self.orientation
        elif self.orientation > math.pi/2:
            angle = math.pi - self.orientation
        elif self.orientation < 0 and self.orientation > -math.pi/2:
            angle = -self.orientation
        else:
            angle = self.orientation + math.pi

        dy = d / math.cos(-angle)

        side = (self.slope * self.points[furthest_point, 0] -
                self.points[furthest_point, 1] + self.intercept)

        if side < 0:
            self.intercept += dy
        else:
            self.intercept -= dy

        self._create_line()

    def target_orientation(self, primary_orientations):
        po_diff = [min_angle_difference(self.orientation, o) for
                   o in primary_orientations]
        min_po_diff = min(po_diff)
        return primary_orientations[po_diff.index(min_po_diff)]

    def regularize(self, slope, max_error=None):
        # https://math.stackexchange.com/questions/1377716/how-to-find-a-least-squares-line-with-a-known-slope
        if not np.isclose(slope, math.tan(self.orientation)):
            self.slope = slope
            self.intercept = (sum(self.points[:, 1] -
                                  self.slope * self.points[:, 0]) /
                              len(self.points))

            if max_error is not None:
                residuals = self.residuals()
                if residuals > max_error:
                    raise ThresholdError("Could not fit a proper line. Error:\
                                         {}".format(residuals))

            self._create_line()

    def change_intercept(self, intercept):
        self.intercept = intercept
        self._create_line()

    def line_intersect(self, line):
        """
        Compute the intersection between this line and another.

        Parameters
        ----------
        line : (1x2) array-like
            The slope and intersect of a line.

        Returns
        -------
        x, y : float
            The coordinates of intersection. Returns False if no intersection
            found.
        """
        line_self = np.array([self.slope, -1, -self.intercept])
        line_other = np.array([line[0], -1, -line[1]])
        d = line_self[0] * line_other[1] - line_self[1] * line_other[0]
        dx = line_self[2] * line_other[1] - line_self[1] * line_other[2]
        dy = line_self[0] * line_other[2] - line_self[2] * line_other[0]
        if d != 0:
            x = dx / float(d)
            y = dy / float(d)
            return x, y
        else:
            return False
