import numpy as np
from scipy import signal


def align2D(cur_img, ref_patch_with_border, ref_patch, 
    n_iter, cur_px_estimate, no_simd=False):
    # 
    half_patch_size = 4
    patch_size = 8
    patch_area = patch_size ** 2
    is_converged = False

    # ref_patch_dx
    # ref_patch_dy

    H = np.zeros((3, 3), dtype=np.float64)

    # calculate gradient and hessian
    ref_step = patch_size + 2
    mask_x, mask_y = np.array([[-1, 0, 1]]), np.array([[-1], [0], [1]])

    dx = 0.5 * signal.correlate2d(ref_patch_with_border[1:-1, :], mask_x, mode='valid')
    dy = 0.5 * signal.correlate2d(ref_patch_with_border[:, 1:-1], mask_y, mode='valid')

    H[0, 0] = (dx ** 2).sum()
    H[0, 1] = (dx * dy).sum()
    H[0, 2] = dx.sum()
    H[1, 0] = H[0, 1]
    H[1, 1] = (dy ** 2).sum()
    H[1, 2] = dy.sum()
    H[2, 0] = H[0, 2]
    H[2, 1] = H[1, 2]
    H[2, 2] = patch_area

    Hinv = np.linalg.inv(H)
    mean_diff = 0

    # calculate position in new image
    u, v = cur_px_estimate

    # termination criterion
    min_update_squared = 0.03 ** 2
    cur_step = cur_img

    update = np.zeros(3, dytpe=np.float64)
    

