import pyxel
import numpy as np
from dataclasses import dataclass
from enum import Enum, auto

from pipeline import *
from geometry import *
from structs import *

# 人間の直感に優しい視野角(60度)
fov_deg = 60


# 実験のモード
class ExperiMode:
    DOTS = auto()
    IMG = auto()


class App:
    def __init__(self):
        pyxel.init(320, 240, title="The Adventure of the Vection")

        # カメラ
        self.camera = Camera(pos=[0, 0, 0])

        # 実験のモード
        self.experi_mode = ExperiMode.DOTS

        # 視野角の計算
        self.focal_length = (pyxel.width / 2) / np.tan(np.deg2rad(fov_deg) / 2.0)

        # 立方体
        self.cube_points = create_cube_points_cloud(side_length=3)

        # 地平線
        self.floor_points = create_grid_horizon_cloud(
            grid_size=20, grid_length=2.0, points_per_edge=10, offset=[0, -2, 0]
        )

        # 学校の廊下の画像
        self.hall_img = pyxel.Image(389, 218)
        self.hall_img.load(0, 0, "imgs/hall.png")
        self.hall_img_point = np.array([[0, 0, 3]])
        self.hall_img_world_width = 4

        pyxel.run(self.update, self.draw)

    def update(self):
        self.camera.pos[2] = pyxel.sin(pyxel.frame_count) * 10 - 8

        if pyxel.btnp(pyxel.KEY_SPACE):
            self.experi_mode = (
                ExperiMode.IMG
                if self.experi_mode == ExperiMode.DOTS
                else ExperiMode.DOTS
            )

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

        # 画像の大きさを計算
        hall_img_point_2d, hall_img_point_2d_mask = world_to_screen(
            self.hall_img_point,
            self.camera,
            self.focal_length,
            pyxel.width,
            pyxel.height,
        )

        if self.experi_mode == ExperiMode.DOTS:
            # それぞれの点群を描画
            render_points(cube_positions_2d, cube_positions_2d_mask)
            # render_points(floor_points_2d, floor_points_2d_mask, col=1)
        else:
            # 廊下の画像を描画
            base_scale = self.focal_length / (
                self.hall_img_point[0, 2] - self.camera.pos[2]
            )
            hall_img_scale = base_scale * (
                self.hall_img_world_width / self.hall_img.width
            )
            print(hall_img_scale)
            if hall_img_point_2d_mask[0]:
                pyxel.blt(
                    hall_img_point_2d[0, 0] - self.hall_img.width / 2,
                    hall_img_point_2d[0, 1] - self.hall_img.height / 2,
                    self.hall_img,
                    0,
                    0,
                    self.hall_img.width,
                    self.hall_img.height,
                    scale=hall_img_scale,
                )


App()
