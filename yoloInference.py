import sys
import torch
import cv2
import numpy as np
from typing import Optional, List, Tuple

from .bbox import BBox


class YoloFuncs:
    def __init__(self, yoloVersion: int = None, imgSize: int = 640):
        # This really probably doesn't belong here, but lazy...
        self._imgSize = imgSize

        if yoloVersion == 8:
            import ultralytics

            def load_wrapper(weights: str, map_location=None, **kwargs):
                model = ultralytics.YOLO(weights, **kwargs)
                if map_location:
                    model.to(map_location)
                return model

            self.load_model = load_wrapper
            self.non_max_suppression = ultralytics.utils.ops.non_max_suppression
            self.scale_coords = ultralytics.utils.ops.scale_coords
            self.run_inference = self._run_inference_v8
        elif yoloVersion == 7:
            import yolov7

            self.load_model = yolov7.models.experimental.attempt_load
            self.non_max_suppression = yolov7.utils.general.non_max_suppression
            self.scale_coords = yolov7.utils.general.scale_coords
            self.run_inference = self._run_inference_v5_v7
        elif yoloVersion == 5:
            import yolov5

            # yolov5 module doesn't seem to have a scale_coords...
            from yolov7.utils.general import (
                scale_coords,
            )

            def load_wrapper(weights: str, map_location=None, **kwargs):
                return yolov5.models.experimental.attempt_load(
                    weights, device=map_location, **kwargs
                )

            self.load_model = load_wrapper
            self.non_max_suppression = yolov5.utils.general.non_max_suppression
            self.scale_coords = scale_coords
            self.run_inference = self._run_inference_v5_v7
        else:
            raise Exception("Invalid YOLO version selected")

    def _run_inference_v5_v7(
        self,
        model,
        labels: List[str],
        img: np.ndarray,
        conf_thresh: float = 0.25,
        device: str = None,
    ) -> list[(BBox, float, int, str)]:
        yoloImg, ratio, (xPad, yPad) = letterbox(img, self._imgSize)

        # BGR to RGB and HWC to CHW
        yoloImg = yoloImg[:, :, ::-1].transpose(2, 0, 1)
        yoloImg = np.ascontiguousarray(yoloImg)
        yoloImg = torch.from_numpy(yoloImg).to(device)
        # uint8 to fp16/32
        yoloImg = yoloImg.float()
        # 0 - 255 to 0.0 - 1.0
        yoloImg /= 255.0
        if yoloImg.ndimension() == 3:
            yoloImg = yoloImg.unsqueeze(0)
        _, _, yoloImgH, yoloImgW = yoloImg.shape
        imgY, imgX, _ = img.shape

        # NMS IOU threshold
        iou_thres = 0.45
        classes = None
        agnostic_nms = False
        max_det = 5000
        preds = model(yoloImg)[0]
        nms_preds = self.non_max_suppression(
            preds, conf_thresh, iou_thres, classes, agnostic_nms
        )

        results = []
        for det in nms_preds:
            # Rescale box coords to img size
            det[:, :4] = self.scale_coords(
                yoloImg.shape[2:], det[:, :4], img.shape
            ).round()
            for x1, y1, x2, y2, conf, objclass in reversed(det):
                bbox = BBox.fromX1Y1X2Y2(
                    x1.cpu(), y1.cpu(), x2.cpu(), y2.cpu(), imgX, imgY
                )
                objclass = int(objclass)
                label = (
                    labels[objclass]
                    if labels is not None and objclass < len(labels)
                    else ""
                )
                results.append((bbox, float(conf), objclass, label))
        return results

    def _run_inference_v8(
        self,
        model,
        labels: List[str],
        img: np.ndarray,
        conf_thresh: float = 0.25,
        device: str = None,
    ) -> list[(BBox, float, int, str)]:
        preds = model.predict(img)
        pred = preds[0]

        results = []
        for box in pred.boxes:
            bbox = BBox.fromRX1Y1X2Y2(*box.xyxyn[0].tolist())
            objclass = int(box.cls)
            model_label = model.names.get(objclass, None)
            label = labels[objclass]
            if model_label and label != model_label:
                pass  # Mismatch between model's label and passed in labels
            results.append((bbox, float(box.conf), objclass, label))
        return results


class YoloInference:
    def __init__(
        self,
        weights: str,
        imgSize: int = 640,
        labels: list[str] = None,
        device: str = "cpu",
        yoloVersion: int = 8,
    ):
        """Constructor for YoloInference

        Parameters:
        weights (str): Path to weights file
        imgSize (int): Resolution model was trained on
        labels (list[str]): class labels, can be None
        device (str): device type to pass to torch (cpu or cuda)
        yoloVersion(int): yolo version to use. Can be 5, 7 or 8
        """
        self._device = torch.device(device)
        self._imgSize = imgSize
        self._labels = labels

        self._yoloFuncs: YoloFuncs = YoloFuncs(
            yoloVersion=yoloVersion, imgSize=self._imgSize
        )
        self._yolo = self._yoloFuncs.load_model(weights, map_location=self._device)

    def getLabel(self, objClass):
        if self._labels and objClass < len(self._labels):
            return self._labels[objClass]
        return ""

    def runInference(
        self, img: np.ndarray, conf_thresh: float = 0.25
    ) -> list[(BBox, float, int, str)]:
        """Runs inference on an image

        Parameters:
        img (np.ndarray): image to run inference on
        conf_thresh (float): minimum confidence result to return

        Returns:
        list[( bbox (BBox), conf (float), class (int), label (str) )]"""

        return self._yoloFuncs.run_inference(
            model=self._yolo,
            labels=self._labels,
            img=img,
            conf_thresh=conf_thresh,
            device=self._device,
        )


def letterbox(
    img,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scaleFill=False,
    scaleup=True,
    stride=32,
):
    """Resize and pad image while meeting stride-multiple constraints"""

    # current shape [height, width]
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    # only scale down, do not scale up (for better test mAP)
    if not scaleup:
        r = min(r, 1.0)

    # Compute padding
    # width, height ratios
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    # wh padding
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        # width, height ratios
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    # divide padding into 2 sides
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return img, ratio, (dw, dh)
