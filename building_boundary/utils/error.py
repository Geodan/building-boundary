# -*- coding: utf-8 -*-
"""

@author: Chris Lucas
"""


class ThresholdError(Exception):
    def __init__(self, msg):
        self.msg = msg
