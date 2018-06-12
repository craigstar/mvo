import numpy as np

from .log import LOG_DEBUG, LOG_INFO, LOG_WARN, LOG_ERROR, LOG_CRITICAL

class Matcher(object):
    """docstring for Matcher"""
    def __init__(self, arg):
        pass

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
                    search_level, half_patch_size, patch):
        patch_size = half_patch_size * 2
        A_ref_cur = np.linalg.inv(A_cur_ref)
        if np.isnan(A)[0, 0]:
            LOG_ERROR("Affine warp is NaN, probably camera has no translation")
            return

        px_ref_pyr = px_ref / (2 ** level_ref)
        for i in range(patch_size):
            