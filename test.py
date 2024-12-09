#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
from typing import List, Tuple, Optional

import cv2
import numpy as np

from yolox.yolox_onnx import YoloxONNX


def main() -> None:
    # 動画読み込み
    cap: cv2.VideoCapture = cv2.VideoCapture("sample.mp4")
    cap_fps: float = cap.get(cv2.CAP_PROP_FPS)

    # モデルロード
    score_th: float = 0.3
    yolox: YoloxONNX = YoloxONNX(
        model_path="model/yolox_nano.onnx",
        class_score_th=score_th,
        nms_th=0.45,
        nms_score_th=0.1,
        with_p6=False,
        providers=["CPUExecutionProvider"],
    )

    # インペインティングマスク
    inpaint_mask: np.ndarray = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE)

    # COCOクラスリスト読み込み
    with open("yolox/coco_classes.txt", "rt") as f:
        coco_classes: List[str] = f.read().rstrip("\n").split("\n")

    video_writer: Optional[cv2.VideoWriter] = None

    while True:
        # カメラキャプチャ
        ret: bool
        frame: Optional[np.ndarray]
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # 推論実施(通常)
        inference_start_time: float = time.time()
        bboxes, scores, class_ids = inference(yolox, frame)
        inference_elapsed_time: float = time.time() - inference_start_time

        # デバッグ描画
        debug_frame1: np.ndarray = copy.deepcopy(frame)
        debug_frame1 = draw_debug(
            debug_frame1,
            "PreProcess: None",
            None,
            inference_elapsed_time,
            score_th,
            bboxes.tolist(),
            scores.tolist(),
            class_ids.tolist(),
            coco_classes,
        )

        # 推論実施(インペインティング(INPAINT_TELEA))
        preprocess_start_time = time.time()
        inpaint_frame: np.ndarray = cv2.inpaint(
            frame, inpaint_mask, 3, cv2.INPAINT_TELEA
        )
        preprocess_elapsed_time = time.time() - preprocess_start_time

        inference_start_time = time.time()
        bboxes, scores, class_ids = inference(yolox, inpaint_frame)
        inference_elapsed_time = time.time() - inference_start_time

        debug_frame2: np.ndarray = copy.deepcopy(inpaint_frame)
        debug_frame2 = draw_debug(
            debug_frame2,
            "PreProcess: Inpaint(INPAINT_TELEA)",
            preprocess_elapsed_time,
            inference_elapsed_time,
            score_th,
            bboxes.tolist(),
            scores.tolist(),
            class_ids.tolist(),
            coco_classes,
        )

        # 比較動画
        debug_image: np.ndarray = cv2.hconcat([debug_frame1, debug_frame2])

        # VideoWriter書き込み
        if video_writer is None:
            writer_width: int = debug_image.shape[1]
            writer_height: int = debug_image.shape[0]
            video_writer = cv2.VideoWriter(
                "output.mp4",
                cv2.VideoWriter_fourcc(*"mp4v"),  # type:ignore
                int(cap_fps),
                (writer_width, writer_height),
            )
        if video_writer is not None:
            video_writer.write(debug_image)

        # キー処理(ESC：終了)
        key: int = cv2.waitKey(1)
        if key == 27:  # ESC
            break

        debug_image = cv2.resize(debug_image, dsize=None, fx=0.5, fy=0.5)
        cv2.imshow("YOLOX ONNX Sample", debug_image)

    if video_writer is not None:
        video_writer.release()
    cap.release()
    cv2.destroyAllWindows()


def inference(
    model: YoloxONNX, image: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    bboxes: np.ndarray
    scores: np.ndarray
    class_ids: np.ndarray

    bboxes, scores, class_ids = model.inference(image)

    target_id: List[int] = [2]
    target_index: np.ndarray = np.in1d(class_ids, np.array(target_id))
    bboxes = bboxes[target_index]
    scores = scores[target_index]
    class_ids = class_ids[target_index]

    return bboxes, scores, class_ids


def draw_debug(
    image: np.ndarray,
    title: str,
    preprocess_elapsed_time: Optional[float],
    inference_elapsed_time: float,
    score_th: float,
    bboxes: List[Tuple[float, float, float, float]],
    scores: List[float],
    class_ids: List[int],
    coco_classes: List[str],
) -> np.ndarray:
    debug_image: np.ndarray = copy.deepcopy(image)

    for bbox, score, class_id in zip(bboxes, scores, class_ids):
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        if score_th > score:
            continue

        debug_image = cv2.rectangle(
            debug_image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            thickness=2,
        )

        score_str: str = "%.2f" % score
        text: str = "%s:%s" % (str(coco_classes[int(class_id)]), score_str)
        debug_image = cv2.putText(
            debug_image,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            thickness=2,
        )

    debug_image = cv2.putText(
        debug_image,
        title,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        thickness=2,
    )

    if preprocess_elapsed_time is None:
        elapsed_time_text: str = "Elapsed time(Preprocess):-"
    else:
        elapsed_time_text = "Elapsed time(Preprocess):" + "%.0f" % (
            preprocess_elapsed_time * 1000
        )
        elapsed_time_text += "ms"

    debug_image = cv2.putText(
        debug_image,
        elapsed_time_text,
        (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        thickness=2,
    )

    elapsed_time_text = "Elapsed time(Inference):" + "%.0f" % (
        inference_elapsed_time * 1000
    )
    elapsed_time_text += "ms"
    debug_image = cv2.putText(
        debug_image,
        elapsed_time_text,
        (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        thickness=2,
    )

    return debug_image


if __name__ == "__main__":
    main()
