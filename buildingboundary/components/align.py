# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""

import numpy as np


def align_by_intersept(segments, max_diff):
    d = {}
    for s in segments:
        if round(s.slope, 4) in d:
            d[round(s.slope, 4)].append(s)
        else:
            d[round(s.slope, 4)] = [s]

    for v in d.values():
        intersepts = np.array([x.intersept for x in v])
        order = np.argsort(intersepts)
        intersepts_sorted = intersepts[order]
        diff_idx = [0]+list(np.where(np.diff(intersepts_sorted) >
                                     max_diff)[0] + 1)+[len(v)]

        clusters = [order[x:y] for x, y in
                    zip(diff_idx[:-1], np.roll(diff_idx, -1)[:-1])]

        for c in clusters:
            if len(c) > 1:
                new_intersept = np.mean(intersepts[c])
                for i in c:
                    v[i].change_intersept(new_intersept)
