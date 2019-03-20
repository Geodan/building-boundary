# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""


def check_error(segments, max_error):
    """
    Checks the error (average distance between the points and
    the fitted line) of the segments.

    Parameters
    ----------
    segments : list of segments
        The wall segments to check.
    max_error : int or float
        The error at which a segments will be classified as invalid.

    Returns
    -------
    invalid_segments : list of int
        The indices of the segments which have a higher error than
        the given max error.
    """
    invalid_segments = []
    for i, s in enumerate(segments):
        error = s.residuals()
        if error > max_error:
            invalid_segments.append(i)
    return invalid_segments


def restore(segments, original_segments, invalid_segments,
            merged_segments, removed_segments):
    """
    Replaces invalid segments with earlier segments.

    Parameters
    ----------
    segments : list of segments
        The current wall segments.
    original_segments : list of segments
        The wall segments before any merging took place.
    invalid_segments : list of int
        The indices of the segments which have a higher error than
        the given max error.
    merged_segments : list of list of int
        The indices of the segments that were merged grouped together
        in lists.
    removed_segments : list of int
        The indices of the segments that got removed.

    Returns
    -------
    segments : list of segments
        The segments with the original segments restored in place of the
        invalid segments.
    """
    merged_segments = [s for i, s in enumerate(merged_segments) if
                       i not in removed_segments]
    offset = 0
    for invalid in invalid_segments:
        restore_segments = [original_segments[i] for
                            i in merged_segments[invalid]]
        segments[invalid+offset:invalid+offset+1] = restore_segments
        offset += len(restore_segments)-1
    return segments
