import pyxel
import numpy as np

positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])


def create_cube_points_cloud(side_length=2.0, points_per_edge=20):
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

    points_3d = np.column_stack((X_arr, Y_arr, Z_arr))

    return points_3d


def transform_positions(pos_array, offset):
    """一配列のみを高速に一括処理"""
    return pos_array + offset


# デカルト系を画面系に変換する
def world_to_screen(points_3d, focal_length, screen_width, screen_height):
    X = points_3d[:, 0]
    Y = points_3d[:, 1]
    Z = points_3d[:, 2]

    # 0割り算を防ぐために、小さな値（epsilon)を足すか、Z > 0でクリッピング
    Z_safe = np.where(Z == 0, 1e-5, Z)

    # 一括で2D座標を計算する
    x_2d = focal_length * (X / Z_safe) + (screen_width / 2)
    y_2d = (screen_height / 2) - focal_length * (Y / Z_safe)

    # (N, 2) の形状にして返却
    # column_stackは、Fortran連続（列方向）に近く、最適化された並びらしい
    # 一番いい感じでメモリに乗っかる
    return np.column_stack((x_2d, y_2d))


# 画面系の座標展を描画する
def render_points(points_2d, col=7):
    points_2d_list = points_2d.tolist()
    for x, y in points_2d_list:
        pyxel.pset(x, y, col)


class App:
    def __init__(self):
        pyxel.init(320, 240, title="The Adventure of the Vection")

        self.camera_pos = [0, 0, 0]
        self.cube_points = create_cube_points_cloud(side_length=5)

        # ボックスをカメラの前方へ、ずらしておく
        self.cube_points[:, 2] += 10.0

        pyxel.run(self.update, self.draw)

    def update(self):
        self.camera_pos = [0, 0, pyxel.sin(pyxel.frame_count) * 5]
        print(self.camera_pos)
        self.cube_points_offset = transform_positions(self.cube_points, self.camera_pos)

    def draw(self):
        pyxel.cls(0)
        positions_2d = world_to_screen(
            self.cube_points_offset,
            200,
            pyxel.width,
            pyxel.height,
        )
        render_points(positions_2d)


App()
