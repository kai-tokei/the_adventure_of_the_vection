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

        # 学校の廊下の画像
        self.hall_img = pyxel.Image(389, 218)
        self.hall_img.load(0, 0, "imgs/hall.png")

        pyxel.run(self.update, self.draw)

    def update(self):
        self.camera.pos[2] = pyxel.sin(pyxel.frame_count) * 10 - 8

    def draw(self):
        pyxel.cls(0)

        # 立方体を描画
        cube_positions_2d, cube_positions_2d_mask = world_to_screen(
            self.cube_points,
            self.camera,
            self.focal_length,
            pyxel.width,
            pyxel.height,
        )

        # グリッドの床を描画
        floor_points_2d, floor_points_2d_mask = world_to_screen(
            self.floor_points,
            self.camera,
            self.focal_length,
            pyxel.width,
            pyxel.height,
        )

        # それぞれの点群を描画
        render_points(floor_points_2d, floor_points_2d_mask, col=1)
        render_points(cube_positions_2d, cube_positions_2d_mask)

        # 廊下の画像を描画
        pyxel.blt(
            (pyxel.width - self.hall_img.width) / 2,
            (pyxel.height - self.hall_img.height) / 2,
            self.hall_img,
            0,
            0,
            self.hall_img.width,
            self.hall_img.height,
            scale=pyxel.sin(pyxel.frame_count) * 2.0,
        )


App()
