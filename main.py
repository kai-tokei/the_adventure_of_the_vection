import pyxel
import numpy as np
from dataclasses import dataclass

from pipeline import *
from geometry import *
from structs import *

# 人間の直感に優しい視野角
fov_deg = 60
fov_rad = np.radians(fov_deg)


class App:
    def __init__(self):
        pyxel.init(320, 240, title="The Adventure of the Vection")

        self.camera = Camera(pos=[0, 0, 0])

        self.focal_length = (pyxel.width / 2) / np.tan(fov_rad / 2.0)

        # 立方体
        self.cube_points = create_cube_points_cloud(side_length=3)

        # 地平線
        self.floor_points = create_grid_horizon_cloud(
            grid_size=20, grid_length=2.0, points_per_edge=10, offset=[0, -2, 0]
        )

        pyxel.run(self.update, self.draw)

    def update(self):
        self.camera.pos[2] = pyxel.sin(pyxel.frame_count) * 10 - 8

    def draw(self):
        pyxel.cls(0)

        cube_positions_2d, cube_positions_2d_mask = world_to_screen(
            self.cube_points,
            self.camera,
            self.focal_length,
            pyxel.width,
            pyxel.height,
        )

        floor_points_2d, floor_points_2d_mask = world_to_screen(
            self.floor_points,
            self.camera,
            self.focal_length,
            pyxel.width,
            pyxel.height,
        )

        render_points(floor_points_2d, floor_points_2d_mask, col=1)
        render_points(cube_positions_2d, cube_positions_2d_mask)


App()
