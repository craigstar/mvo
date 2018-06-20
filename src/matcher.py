import numpy as np

from .log import LOG_DEBUG, LOG_INFO, LOG_WARN, LOG_ERROR, LOG_CRITICAL

class Matcher(object):
    """docstring for Matcher"""
    def __init__(self, arg):
        self.half_patch_size = 4
        self.patch_size = 8
        self.patch_with_border = None

    def create_patch_from_patch_with_border(self):
        start = (10 - self.patch_size) // 2
        end = 10 - start
        self.patch = self.patch_with_border[start:end, start:end]

    def get_warp_matrix_affine(self, cam_ref, cam_cur,
                               px_ref, f_ref, depth_ref,
                               T_cur_ref, level_ref):
        half_patch_size = 5
        xyz_ref = f_ref * depth_ref

        xyz_du_ref = cam_ref.cam2world(px_ref + np.array([half_patch_size, 0]) * (2 ** level_ref))
        xyz_dv_ref = cam_ref.cam2world(px_ref + np.array([0, half_patch_size]) * (2 ** level_ref))
        xyz_du_ref *= xyz_ref[2] / xyz_du_ref[2]
        xyz_dv_ref *= xyz_ref[2] / xyz_dv_ref[2]
        px_cur = cam_cur.world2cam(T_cur_ref * xyz_ref)
        px_du = cam_cur.world2cam(T_cur_ref * xyz_du_ref)
        px_dv = cam_cur.world2cam(T_cur_ref * xyz_dv_ref)
        A_cur_ref = np.empty((2,2))
        A_cur_ref[:, 0] = (px_du - px_cur) / half_patch_size
        A_cur_ref[:, 1] = (px_dv - px_cur) / half_patch_size
        return A_cur_ref

    def get_best_search_level(self, A_cur_ref, max_level):
        # compute pyramid level in other images
        search_level = 0
        D = np.linalg.det(A_cur_ref)
        while D > 3 and search_level < max_level:
            search_level += 1
            D *= 0.25
        return search_level

    def warp_affine(self, A_cur_ref, img_ref, px_ref, level_ref,
                    search_level, half_patch_size):
        cols, rows, _* = img_ref.shape
        patch_size = half_patch_size * 2
        A_ref_cur = np.linalg.inv(A_cur_ref)
        if np.isnan(A)[0, 0]:
            LOG_ERROR("Affine warp is NaN, probably camera has no translation")
            return

        patch = np.zeros((patch_size, patch_size), dtype=np.uint8)      # image pixel
        px_ref_pyr = px_ref / (2 ** level_ref)
        for y in range(patch_size):
            for x in range(patch_size):
                px_patch = np.array([x - half_patch_size, y - half_patch_size])
                px_patch *= (2 ** search_level)
                px_x, px_y = A_ref_cur.dot(px_patch) + px_ref_pyr
                if not (px_x < 0 or px_x >= cols - 1 or
                        px_y < 0 or px_y >= rows - 1):
                    patch[x, y] = self.interpolate(img_ref, px_x, px_y)
        return patch

    def interpolate(self, img, u, v):
        x, y = int(u), int(v)
        subpix_x, subpix_y = u - x, v - y

        w00 = (1.0 - subpix_x) * (1.0 - subpix_y)
        w01 = (1.0 - subpix_x) * subpix_y
        w10 = subpix_x * (1.0 - subpix_y)
        w11 = 1.0 - w00 - w01 - w10
        w = np.array([[w00, w10], [w01, w11]])
        
        patch = img[y:y+2, x:x+2]
        return (patch * w).sum()