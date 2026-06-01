import pyxel
import numpy as np
import cv2
import mediapipe as mp
import time
from dataclasses import dataclass
from enum import Enum, auto

from pipeline import *
from geometry import *
from structs import *

# 人間の直感に優しい視野角(60度)
fov_deg = 60

REAL_EYE_DISTANCE = 63.0  # 両眼の感覚（人間の平均は63mm）
FOCAL_LENGTH_PC_CAMERA = 500.0  # 焦点距離
LERP_FACTOR = 0.15
DETECTION_TIME_DISTANCE = 4

# MediaPipeの設定
BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode


# 実験のモード
class ExperiMode:
    DOTS = auto()
    IMG = auto()


class App:
    def __init__(self):
        pyxel.init(320, 240, title="The Adventure of the Vection", fps=30)
        pyxel.mouse(True)

        # カメラ
        self.camera = Camera(pos=[0, 0, 0])
        self.camera_target_z = 0

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

        # 廊下（点群）
        self.hall_points = np.concatenate(
            [
                create_cube_points_cloud(
                    side_length=3, offset=np.array([0, 0, 3 + i * 3])
                )
                for i in range(9)
            ]
        )

        # 学校の廊下の画像
        self.hall_img = pyxel.Image(389, 218)
        self.hall_img.load(0, 0, "imgs/hall.png")
        self.hall_img_point = np.array([[0, -1, 27]])
        self.hall_img_world_width = 64

        # カメラと顔検出機の初期化
        options = FaceDetectorOptions(
            base_options=BaseOptions(
                model_asset_path="face_models/blaze_face_short_range.tflite"
            ),
            running_mode=VisionRunningMode.VIDEO,  # ビデオストリームモード
        )

        self.detector = FaceDetector.create_from_options(options)

        self.cap = cv2.VideoCapture(0)

        pyxel.run(self.update, self.draw)

    def update(self):
        # スペースキーでモードを切り替え
        if pyxel.btnp(pyxel.KEY_SPACE):
            self.experi_mode = (
                ExperiMode.IMG
                if self.experi_mode == ExperiMode.DOTS
                else ExperiMode.DOTS
            )

        if (pyxel.frame_count % DETECTION_TIME_DISTANCE == 0) and self.cap.isOpened():
            # カメラから画像を取得
            success, frame = self.cap.read()
            if not success:
                return

            # 画像をRGBに変換し、MediaPipe用に整形
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # AI検出を実行
            timestamp_ms = int(time.time() * 1000)
            detection_result = self.detector.detect_for_video(mp_image, timestamp_ms)

            # 顔が検出されたら、顔までの距離を計算
            if detection_result.detections:
                detection = detection_result.detections[0]
                keypoints = detection.keypoints

                if len(keypoints) >= 2:
                    h, w, _ = frame.shape

                    # keypointsとは、コンピュータビジョンにおける特徴点
                    # 0: 左目, 1: 右目, 2: 鼻, 3: 口 ...
                    left_eye = np.array([keypoints[0].x * w, keypoints[0].y * h])
                    right_eye = np.array([keypoints[1].x * w, keypoints[1].y * h])

                    # 画面上の距離を計算
                    pixel_distance = np.linalg.norm(left_eye - right_eye)

                    if pixel_distance > 0:
                        distance_mm = (
                            REAL_EYE_DISTANCE * FOCAL_LENGTH_PC_CAMERA / pixel_distance
                        )
                        distance_cm = distance_mm / 10.0

                        # print(distance_cm, 15 - distance_cm * 0.5)

                        # カメラの座標を設定
                        self.camera_target_z = 15 - distance_cm * 0.5

        # カメラを目標座標に向かって線型補完しながら滑らかに移動させる
        self.camera.pos[2] += (self.camera_target_z - self.camera.pos[2]) * LERP_FACTOR

        # デバッグ用
        # self.camera.pos[2] = pyxel.sin(pyxel.frame_count) * 10 - 4

    def draw(self):
        pyxel.cls(0)

        # 立方体を座標計算
        # cube_positions_2d, cube_positions_2d_mask = world_to_screen(
        #     self.cube_points,
        #     self.camera,
        #     self.focal_length,
        #     pyxel.width,
        #     pyxel.height,
        # )

        # グリッドの床を座標計算
        #     floor_points_2d, floor_points_2d_mask = world_to_screen(
        #     self.floor_points,
        #     self.camera,
        #     self.focal_length,
        #     pyxel.width,
        #     pyxel.height,
        # )

        if self.experi_mode == ExperiMode.DOTS:
            # 廊下の点群を座標計算
            hall_points_2d, hall_points_2d_mask = world_to_screen(
                self.hall_points,
                self.camera,
                self.focal_length,
                pyxel.width,
                pyxel.height,
            )

            # それぞれの点群を描画
            render_points(hall_points_2d, hall_points_2d_mask)
            # render_points(cube_positions_2d, cube_positions_2d_mask)
            # render_points(floor_points_2d, floor_points_2d_mask, col=1)
        else:
            # 廊下の画像を座標計算
            hall_img_point_2d, hall_img_point_2d_mask = world_to_screen(
                self.hall_img_point,
                self.camera,
                self.focal_length,
                pyxel.width,
                pyxel.height,
            )

            # 廊下の画像を描画
            base_scale = self.focal_length / (
                self.hall_img_point[0, 2] - self.camera.pos[2]
            )
            hall_img_scale = base_scale * (
                self.hall_img_world_width / self.hall_img.width
            )
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
