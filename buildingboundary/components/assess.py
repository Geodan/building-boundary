# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""


def check_error(segments, max_error):
    invalid_segments = []
    for i, s in enumerate(segments):
        error = s.residuals()
        if error > max_error:
            invalid_segments.append(i)
    return invalid_segments


def restore(segments, original_segments, invalid_segments,
            merged_segments, removed_segments):
    merged_segments = [s for i, s in enumerate(merged_segments) if
                       i not in removed_segments]
    offset = 0
    for invalid in invalid_segments:
        restore_segments = [original_segments[i] for
                            i in merged_segments[invalid]]
        segments[invalid+offset:invalid+offset+1] = restore_segments
        offset += len(restore_segments)-1
    return segments
