import numpy as np
import copy

from .config import Config

class Reprojector(object):
    """docstring for Reprojector"""

    class Candidate(object):
        """docstring for Candidate"""
        def __init__(self, point, uv):
            self.pt = point
            self.uv = uv

    class Grid(object):
        """docstring for Grid"""
        def __init__(self):
            self.cell_size = 0
            self.grid_cols = 0
            self.grid_rows = 0
            self.cells = None
            self.cell_order = None

    class Options(object):
        """docstring for Options"""
        def __init__(self):
            self.max_n_kfs = 10
            self.find_match_direct = True
            
            
    def __init__(self, cam, pt_map):
        self.cam = cam
        self.map = pt_map
        self.grid = Reprojector.Grid()
        self.options = Reprojector.Options()
        # self.cell = []
        # self.candidate_grid = []
        self.initialize_grid(cam)

    def _reset_grid(self):
        pass

    def initialize_grid(self, cam):
        self.grid.cell_size = Config.get_grid_size()
        self.grid.cols = int(self.cam.width / self.grid.cell_size)
        self.grid.rows = int(self.cam.height / self.grid.cell_size)

        # TODO:
        self.grid.cells = [[] for i in range(self.grid.cols * self.grid.rows)]
        self.grid.cell_order = np.arange(len(self.grid.cells))
        np.random.shuffle(self.grid.cell_order)

    def reproject_point(self, frame, point3d):
        uv_cur = frame.w2c(point3d.pos.copy())
        if frame.cam.is_in_frame(uv_cur, 8):
            k = int(uv_cur[1] // self.grid.cell_size * self.grid.cols
                  + uv_cur[0] // self.grid.cell_size)
            self.grid.cells[k].append(Reprojector.Candidate(point3d, uv_cur))
            return True
        return False

    def reproject_map(self, frame):
        self._reset_grid()
        overlap_kfs = {}

        close_kfs = self.map.get_close_keyframes(frame)
        close_kfs.sort(key=lambda e: e[1])              # (frame, distance) pairs, sort by distance

        for i, frm in enumerate(close_kfs):
            if i >= self.options.max_n_kfs:
                break

            frm_ref, d = frm
            overlap_kfs[frm_ref] = 0

            for ft in frm_ref.features:
                # to check if this 2d point got a 3d map point
                if ft.point is None:
                    continue

                # make sure one 3d point projects only once
                if ft.point.last_project_kf_id == frame.id:
                    continue
                ft.point.last_project_kf_id = frame.id

                if self.reproject_point(frm_ref, ft.point):
                    overlap_kfs[frm_ref] += 1
        return overlap_kfs