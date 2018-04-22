import sophus as sp


class SE3(sp.SE3):
    """Improvment for SE3"""
    def __init__(self, *args):
        if len(args) == 2:
            R, t = args
            args = np.eye(4)
            args[:3, :3] = R
            args[:2, 3] = t
        print('args is', args)
        super(SE3, self).__init__(*args)
