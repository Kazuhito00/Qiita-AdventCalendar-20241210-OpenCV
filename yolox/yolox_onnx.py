#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
from typing import List, Tuple, Optional

import cv2
import numpy as np
import onnxruntime  # type:ignore


class YoloxONNX:
    def __init__(
        self,
        model_path: str = "yolox_nano.onnx",
        class_score_th: float = 0.3,
        nms_th: float = 0.45,
        nms_score_th: float = 0.1,
        with_p6: bool = False,
        providers: List[str] = ["CUDAExecutionProvider", "CPUExecutionProvider"],
    ) -> None:
        # 閾値
        self.class_score_th: float = class_score_th
        self.nms_th: float = nms_th
        self.nms_score_th: float = nms_score_th

        self.with_p6: bool = with_p6

        # モデル読み込み
        self.onnx_session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(
            model_path,
            providers=providers,
        )

        self.input_detail = self.onnx_session.get_inputs()[0]
        self.input_name: str = self.input_detail.name
        self.output_name: str = self.onnx_session.get_outputs()[0].name

        # 各種設定
        self.input_shape: Tuple[int, int] = self.input_detail.shape[2:]

    def inference(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        temp_image: np.ndarray = copy.deepcopy(image)
        image_height, image_width = image.shape[0], image.shape[1]

        # 前処理
        image, ratio = self._preprocess(temp_image, self.input_shape)

        # 推論実施
        results: List[np.ndarray] = self.onnx_session.run(
            None,
            {self.input_name: image[None, :, :, :]},
        )

        # 後処理
        bboxes, scores, class_ids = self._postprocess(
            results[0],
            self.input_shape,
            ratio,
            self.nms_th,
            self.nms_score_th,
            image_width,
            image_height,
            p6=self.with_p6,
        )

        return bboxes, scores, class_ids

    def _preprocess(
        self,
        image: np.ndarray,
        input_size: Tuple[int, int],
        swap: Tuple[int, int, int] = (2, 0, 1),
    ) -> Tuple[np.ndarray, float]:
        if len(image.shape) == 3:
            padded_image: np.ndarray = (
                np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
            )
        else:
            padded_image = np.ones(input_size, dtype=np.uint8) * 114

        ratio: float = min(
            input_size[0] / image.shape[0], input_size[1] / image.shape[1]
        )
        resized_image: np.ndarray = cv2.resize(
            image,
            (int(image.shape[1] * ratio), int(image.shape[0] * ratio)),
            interpolation=cv2.INTER_LINEAR,
        )
        resized_image = resized_image.astype(np.uint8)

        padded_image[: int(image.shape[0] * ratio), : int(image.shape[1] * ratio)] = (
            resized_image
        )
        padded_image = padded_image.transpose(swap)
        padded_image = np.ascontiguousarray(padded_image, dtype=np.float32)

        return padded_image, ratio

    def _postprocess(
        self,
        outputs: np.ndarray,
        img_size: Tuple[int, int],
        ratio: float,
        nms_th: float,
        nms_score_th: float,
        max_width: int,
        max_height: int,
        p6: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        grids: List[np.ndarray] = []
        expanded_strides: List[np.ndarray] = []

        strides: List[int] = [8, 16, 32] if not p6 else [8, 16, 32, 64]

        hsizes: List[int] = [img_size[0] // stride for stride in strides]
        wsizes: List[int] = [img_size[1] // stride for stride in strides]

        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid: np.ndarray = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))

        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)
        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

        predictions: np.ndarray = outputs[0]
        boxes: np.ndarray = predictions[:, :4]
        scores: np.ndarray = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy: np.ndarray = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
        boxes_xyxy /= ratio

        dets: Optional[np.ndarray] = self._multiclass_nms(
            boxes_xyxy,
            scores,
            nms_thr=nms_th,
            score_thr=nms_score_th,
        )

        bboxes: np.ndarray = np.array([])
        scores_out: np.ndarray = np.array([])
        class_ids: np.ndarray = np.array([])
        if dets is not None:
            bboxes, scores_out, class_ids = dets[:, :4], dets[:, 4], dets[:, 5]
            for bbox in bboxes:
                bbox[0] = max(0, bbox[0])
                bbox[1] = max(0, bbox[1])
                bbox[2] = min(bbox[2], max_width)
                bbox[3] = min(bbox[3], max_height)

        return bboxes, scores_out, class_ids

    def _nms(self, boxes: np.ndarray, scores: np.ndarray, nms_thr: float) -> List[int]:
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

        areas: np.ndarray = (x2 - x1 + 1) * (y2 - y1 + 1)
        order: np.ndarray = scores.argsort()[::-1]

        keep: List[int] = []
        while order.size > 0:
            i: int = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= nms_thr)[0]
            order = order[inds + 1]

        return keep

    def _multiclass_nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        nms_thr: float,
        score_thr: float,
        class_agnostic: bool = True,
    ) -> Optional[np.ndarray]:
        if class_agnostic:
            nms_method = self._multiclass_nms_class_agnostic
        else:
            nms_method = self._multiclass_nms_class_aware

        return nms_method(boxes, scores, nms_thr, score_thr)

    def _multiclass_nms_class_aware(
        self, boxes: np.ndarray, scores: np.ndarray, nms_thr: float, score_thr: float
    ) -> Optional[np.ndarray]:
        final_dets: List[np.ndarray] = []
        num_classes: int = scores.shape[1]

        for cls_ind in range(num_classes):
            cls_scores: np.ndarray = scores[:, cls_ind]
            valid_score_mask: np.ndarray = cls_scores > score_thr

            if valid_score_mask.sum() == 0:
                continue

            valid_scores: np.ndarray = cls_scores[valid_score_mask]
            valid_boxes: np.ndarray = boxes[valid_score_mask]
            keep: List[int] = self._nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds: np.ndarray = np.ones((len(keep), 1)) * cls_ind
                dets: np.ndarray = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds],
                    1,
                )
                final_dets.append(dets)

        if len(final_dets) == 0:
            return None

        return np.concatenate(final_dets, 0)

    def _multiclass_nms_class_agnostic(
        self, boxes: np.ndarray, scores: np.ndarray, nms_thr: float, score_thr: float
    ) -> Optional[np.ndarray]:
        cls_inds: np.ndarray = scores.argmax(1)
        cls_scores: np.ndarray = scores[np.arange(len(cls_inds)), cls_inds]

        valid_score_mask: np.ndarray = cls_scores > score_thr

        if valid_score_mask.sum() == 0:
            return None

        valid_scores: np.ndarray = cls_scores[valid_score_mask]
        valid_boxes: np.ndarray = boxes[valid_score_mask]
        valid_cls_inds: np.ndarray = cls_inds[valid_score_mask]
        keep: List[int] = self._nms(valid_boxes, valid_scores, nms_thr)

        dets: Optional[np.ndarray] = None
        if keep:
            dets = np.concatenate(
                [
                    valid_boxes[keep],
                    valid_scores[keep, None],
                    valid_cls_inds[keep, None],
                ],
                1,
            )

        return dets
