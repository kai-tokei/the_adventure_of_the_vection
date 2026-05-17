import pyxel
import numpy as np
from dataclasses import dataclass

# 人間の直感に優しい視野角
fov_deg = 60
fov_rad = np.radians(fov_deg)


def create_grid_horizon_cloud(
    grid_size=30, grid_length=5.0, points_per_edge=10, offset=[0, 0, 0]
):
    """
    グリッドの地平線を描画する
    """
    # グリッド全体の幅
    total_half = (grid_size * grid_length) / 2.0

    # 1つのマスの辺を構成する直線の等分割データ
    t = np.linspace(0, grid_length, points_per_edge)
    ones = np.ones_like(t)

    # グリッドの各線の位置
    lines = np.linspace(-total_half, total_half, grid_size + 1)

    X_list, Y_list, Z_list = [], [], []

    # --- 横方向の線 ---
    for z in lines:
        for x in lines[:-1]:
            X_list.append(x + t)
            Y_list.append(np.zeros_like(t))
            Z_list.append(z * ones)

    # --- 縦方向の線 ---
    for x in lines:
        for z in lines[:-1]:
            X_list.append(x * ones)
            Y_list.append(np.zeros_like(t))
            Z_list.append(z + t)

    X_arr = np.concatenate(X_list)
    Y_arr = np.concatenate(Y_list)
    Z_arr = np.concatenate(Z_list)

    points_3d = np.column_stack((X_arr, Y_arr, Z_arr)) + np.array(offset)

    return points_3d


def create_cube_points_cloud(
    side_length=2.0, points_per_edge=20, offset=np.array([0, 0, 0])
):
    """
    ワイヤーフレーム立方体を構成する3D点群を生成する
    """
    h = side_length / 2.0  # 原点(0, 0, 0)を中心にするための半分の長さ

    # 1つの辺の上の分割座標
    t = np.linspace(-h, h, points_per_edge)
    ones = np.ones_like(t)

    # 立方体の12の辺の座標を定義するためのパターンリスト
    edge_patterns = [
        # Z軸に並行な4本
        (-h * ones, -h * ones, t),
        (-h * ones, h * ones, t),
        (h * ones, -h * ones, t),
        (h * ones, h * ones, t),
        # Y軸に並行な4本
        (-h * ones, t, -h * ones),
        (-h * ones, t, h * ones),
        (h * ones, t, -h * ones),
        (h * ones, t, h * ones),
        # X軸に並行な4本
        (t, -h * ones, -h * ones),
        (t, -h * ones, h * ones),
        (t, h * ones, -h * ones),
        (t, h * ones, h * ones),
    ]

    # 全ての辺の配列を結合する
    X_arr = np.concatenate([p[0] for p in edge_patterns])
    Y_arr = np.concatenate([p[1] for p in edge_patterns])
    Z_arr = np.concatenate([p[2] for p in edge_patterns])

    points_3d = np.column_stack((X_arr, Y_arr, Z_arr)) + offset

    return points_3d


def transform_positions(pos_array, offset):
    """一配列のみを高速に一括処理"""
    return pos_array + offset


# デカルト系を画面系に変換する
def world_to_screen(points_3d, camera, focal_length, screen_width, screen_height):
    X = points_3d[:, 0] - camera.pos[0]
    Y = points_3d[:, 1] - camera.pos[1]
    Z = points_3d[:, 2] - camera.pos[2]

    # カメラより前方にある点だけを有効にする
    valid_mask = Z > 0.1

    # 0割り算を防ぐために、小さな値（epsilon)を足すか、Z > 0でクリッピング
    Z_safe = np.where(valid_mask, Z, 0.1)

    # 一括で2D座標を計算する
    x_2d = focal_length * (X / Z_safe) + (screen_width / 2)
    y_2d = (screen_height / 2) - focal_length * (Y / Z_safe)

    # (N, 2) の形状にして返却
    # column_stackは、Fortran連続（列方向）に近く、最適化された並びらしい
    # 一番いい感じでメモリに乗っかる
    return np.column_stack((x_2d, y_2d)), valid_mask


# 画面系の座標展を描画する
def render_points(points_2d, mask_list, col=7):
    points_2d_list = points_2d.tolist()
    for (x, y), is_valid in zip(points_2d_list, mask_list):
        if is_valid:
            pyxel.pset(x, y, col)


@dataclass
class Camera:
    pos: list


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
