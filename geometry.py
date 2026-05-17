import numpy as np


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
