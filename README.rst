=================
Building Boundary
=================

Trace the boundary of a set of 2D points.

Parameters
----------
points : (Mx2) array
    The coordinates of the points
k : int
    The amount of nearest neighbors used in the concave hull algorithm
max_error : float
    The maximum average error (distance) the points make with the fitted line
merge_angle : float
    The angle (in radians) difference within two segments will be merged
num_points : int, optional
    The number of points a segment needs to be supported by to be considered
    a primary orientation. Will be ignored if primary orientations are set
    manually.
max_intersect_distance : float, optional
    The maximum distance an found intersection can be from the segments.
alignment : float, optional
    If set segments will be aligned (their intercept be set equal by averaging)
    if the difference between their x-intercepts is within this number.
primary_orientations : list of floats, optional
    The desired primary orientations (in radians) of the boundary. If set manually
    here these orientations will not be computed.
inflate : bool
    If set to true the fit lines will be moved to the furthest outside point.

Returns
-------
: (Mx2) array
    The vertices of the computed boundary line