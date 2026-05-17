import numpy as np
import pyxel


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
