import numpy as np

import sophus


class SE3(sophus.SE3):
    """Improvment for SE3"""
    def __new__(cls, *args, **kwargs):
        # Accept R, t as input as well
        if len(args) == 2:
            R, t = args
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t.flatten()
            args = (T,)
        return super().__new__(cls, *args, **kwargs)