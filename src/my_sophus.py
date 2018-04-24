import numpy as np

import sophus

np.set_printoptions(suppress=True)

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
        return super(SE3, cls).__new__(cls, *args, **kwargs)


    def __mul__(self, other):
        # Accept 3d vec as input
        if isinstance(other, np.ndarray):
            if other.ndim == 1:
                other = np.append(other.flatten(), [1])
                return np.matmul(self.matrix()[:3], other)
            else:
                other = np.hstack((other, np.ones((len(other), 1)))).T
                return np.matmul(self.matrix()[:3], other).T

        return SE3(super(SE3, self).__mul__(other).matrix())

    def transByX(self, x):
        """translate by x value in x-axis"""
        mat = self.matrix()
        mat[0, 3] += x
        return SE3(mat)

    def transByY(self, y):
        """translate by y value in y-axis"""
        mat = self.matrix()
        mat[1, 3] += y
        return SE3(mat)

    def transByZ(self, z):
        """translate by z value in z-axis"""
        mat = self.matrix()
        mat[2, 3] += z
        return SE3(mat)

    def transBy(self, *args):
        """translate by x, y, z"""
        if len(args) == 1:
            x, y, z = args[0]
        else:
            x, y, z = args

        mat = self.matrix()
        mat[0, 3] += x
        mat[1, 3] += y
        mat[2, 3] += z
        return SE3(mat)

    def transX(self, x):
        """translate x value in x-axis"""
        mat = self.matrix()
        mat[0, 3] = x
        return SE3(mat)

    def transY(self, y):
        """translate y value in y-axis"""
        mat = self.matrix()
        mat[1, 3] = y
        return SE3(mat)

    def transZ(self, z):
        """translate z value in z-axis"""
        mat = self.matrix()
        mat[2, 3] = z
        return SE3(mat)

    def trans(self, *args):
        """translate by x, y, z"""
        if len(args) == 1:
            x, y, z = args[0].flatten()
        else:
            x, y, z = args
            
        mat = self.matrix()
        mat[0, 3] = x
        mat[1, 3] = y
        mat[2, 3] = z
        return SE3(mat)

    def inverse(self):
        """Return inverse of T"""
        mat = super(SE3, self).inverse().matrix()
        return SE3(mat) 


if __name__ == '__main__':
    """some unit test"""
    R = np.array([[0.707107,  0.707107,       0],
                  [-0.707107, 0.707107,       0],
                  [        0,        0,       1]])
    t = np.array([-1.41421, 0, 0])

    pt0 = np.array([1, 0, 0])
    T = SE3(R, t)
    pt1 = T * pt0

    # print(T)
    # print(pt1)

    t1 = np.array([10, 20, 20])
    # print(T.trans(t1))

    # print(T.trans(10, 20, 20))
    # print(T.transBy(10, 20, 20))

    pts = np.arange(15).reshape((-1, 3))
    pts_new = T * pts
    print(pts_new)
