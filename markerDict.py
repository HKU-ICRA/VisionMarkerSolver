import numpy as np

class MarkerDict:

    def __init__(self):
        self.markers = {
            'one':  [
                        True, True, True, True, True, True, True,
                        True, True, False, False, True, True, True,
                        True, False, True, False, True, True, True,
                        True, True, True, False, True, True, True,
                        True, True, True, False, True, True, True,
                        True, False, False, False, False, False, True,
                        True, True, True, True, True, True, True
                    ],
            'two':  [
                        True, True, True, True, True, True, True,
                        True, True, False, False, False, True, True,
                        True, False, True, True, True, False, True,
                        True, True, True, True, False, True, True,
                        True, True, True, False, True, True, True,
                        True, False, False, False, False, False, True,
                        True, True, True, True, True, True, True
                    ],
            'three':[
                        True, True, True, True, True, True, True,
                        True, True, False, False, False, True, True,
                        True, False, True, True, True, False, True,
                        True, True, True, False, False, True, True,
                        True, False, True, True, True, False, True,
                        True, True, False, False, False, True, True,
                        True, True, False, False, False, True, True
                    ],
            'four': [
                        True, True, True, True, True, True, True,
                        True, True, True, True, False, True, True,
                        True, True, True, False, False, True, True,
                        True, True, False, True, False, True, True,
                        True, False, False, False, False, False, True,
                        True, True, True, True, False, True, True,
                        True, True, True, True, True, True, True
                    ],
            'zero': [
                        True, True, True, True, True, True, True,
                        True, True, False, False, False, True, True,
                        True, False, False, True, True, False, True,
                        True, False, True, False, True, False, True,
                        True, False, True, True, False, False, True,
                        True, True, False, False, False, True, True,
                        True, True, True, True, True, True, True
                    ],
            'v':    [
                        True, True, True, True, True, True, True,
                        True, False, True, True, True, False, True,
                        True, False, True, True, True, False, True,
                        True, False, True, True, True, False, True,
                        True, True, False, True, False, True, True,
                        True, True, True, False, True, True, True,
                        True, True, True, True, True, True, True
                    ],
            'x':    [
                        True, True, True, True, True, True, True,
                        True, False, True, True, True, False, True,
                        True, True, False, True, False, True, True,
                        True, True, True, False, True, True, True,
                        True, True, False, True, False, True, True,
                        True, False, True, True, True, False, True,
                        True, True, True, True, True, True, True
                    ],
            'heart':[
                        True, True, True, True, True, True, True,
                        True, True, False, True, False, True, True,
                        True, False, False, False, False, False, True,
                        True, False, False, False, False, False, True,
                        True, True, False, False, False, True, True,
                        True, True, True, False, True, True, True,
                        True, True, True, True, True, True, True
                    ],
            'farm': [
                        True, True, True, True, True, True, True,
                        True, False, False, False, False, False, True,
                        True, False, True, False, True, False, True,
                        True, False, False, False, False, False, True,
                        True, False, True, False, True, False, True,
                        True, False, False, False, False, False, True,
                        True, True, True, True, True, True, True
                    ]
        }
