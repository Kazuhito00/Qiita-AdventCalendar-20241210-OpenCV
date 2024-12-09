#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np


def mouse_callback(
    event: int, x: int, y: int, flags: int, param: Dict[str, Optional[List[int]]]
) -> None:
    param["mouse_point"] = [x, y]
    if event != 0:
        param["mouse_event"] = event  # type: ignore


def main() -> None:
    """
    マスク生成メイン処理
    """
    # 動画読み込み
    cap: cv2.VideoCapture = cv2.VideoCapture("sample.mp4")
    _, original_frame = cap.read()

    # マウスコールバック設定
    window_name: str = "Create Mask"
    mouse_param: Dict[str, Optional[List[int]]] = {
        "mouse_point": [0, 0],
        "mouse_event": None,
    }
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback, mouse_param)  # type:ignore

    # マウスクリックポイント保持用変数
    mouse_click_history: List[Tuple[int, int]] = []

    # マスク画像
    image_height: int = original_frame.shape[0]
    image_width: int = original_frame.shape[1]
    mask_image: np.ndarray = np.zeros((image_height, image_width), np.uint8)

    while True:
        # フレーム読み込み
        frame: np.ndarray = copy.deepcopy(original_frame)
        debug_image: np.ndarray = copy.deepcopy(frame)
        temp_mask_image: np.ndarray = copy.deepcopy(mask_image)

        # マウスイベント処理
        save_flag: bool = False
        if mouse_param["mouse_event"] is not None:
            # 点設置
            if mouse_param["mouse_event"] == cv2.EVENT_LBUTTONDOWN:
                mouse_click_history.append(tuple(mouse_param["mouse_point"]))  # type: ignore
                save_flag = True
            # アンドゥ
            if mouse_param["mouse_event"] == cv2.EVENT_RBUTTONDOWN:
                if len(mouse_click_history) > 0:
                    mouse_click_history.pop()
                    save_flag = True

        # マスク画像生成
        if len(mouse_click_history) >= 3:
            temp_mask_image = cv2.drawContours(
                temp_mask_image, [np.array(mouse_click_history)], 0, (255, 255, 255), -1
            )

        # デバッグ描画
        debug_image = draw_debug_info(debug_image, mouse_click_history)

        # 画面描画
        cv2.imshow(window_name, debug_image)
        cv2.imshow("Mask", temp_mask_image)
        key: int = cv2.waitKey(1)
        if key == 27:  # ESC
            break

        # マウス操作毎に保存
        if save_flag:
            cv2.imwrite("mask.png", temp_mask_image)
            mouse_param["mouse_event"] = None
    cap.release()
    cv2.destroyAllWindows()


def draw_debug_info(
    debug_image: np.ndarray,
    mouse_click_history: List[Tuple[int, int]],
) -> np.ndarray:
    # クリックポイント
    first_point: Optional[Tuple[int, int]] = None
    prev_point: Optional[Tuple[int, int]] = None

    for index, point in enumerate(mouse_click_history):
        if first_point is None:
            first_point = point

        # 点を描画
        cv2.circle(debug_image, point, 2, (0, 255, 0), thickness=-1)

        # 接続線を描画
        if prev_point is not None:
            cv2.line(
                debug_image,
                prev_point,
                point,
                (0, 255, 0),
                thickness=1,
                lineType=cv2.LINE_AA,
            )
        if index == len(mouse_click_history) - 1 and first_point is not None:
            cv2.line(
                debug_image,
                first_point,
                point,
                (0, 255, 0),
                thickness=1,
                lineType=cv2.LINE_AA,
            )

        prev_point = point

    return debug_image


if __name__ == "__main__":
    main()
