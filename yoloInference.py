import sys
import torch
import cv2
import numpy as np
import os
# requires 'yolov5' to be in in path, https://github.com/ultralytics/yolov5.git
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords

class YoloInference:
    def __init__(self, weights):
        self._device = torch.device('cpu')
        self._yolo = attempt_load( weights, map_location=self._device )
        self._yolo.eval()


    def runInference(self, img, conf_thresh = 0.25):
        yoloImg, ratio, (width,height) = letterbox( img )

        yoloImg = yoloImg[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB and HWC to CHW
        yoloImg = np.ascontiguousarray(yoloImg)
        yoloImg = torch.from_numpy(yoloImg).to(self._device)
        yoloImg = yoloImg.float()  # uint8 to fp16/32
        yoloImg /= 255.0  # 0 - 255 to 0.0 - 1.0
        if yoloImg.ndimension() == 3:
            yoloImg = yoloImg.unsqueeze(0)
        _,_,yoloImgH,yoloImgW = yoloImg.shape

        iou_thres=0.45  # NMS IOU threshold
        classes = None
        agnostic_nms = False
        max_det = 5000
        preds = self._yolo(yoloImg)[0]
        nms_preds = non_max_suppression( preds, conf_thresh, iou_thres, classes, agnostic_nms, max_det=max_det)

        results = []
        for det in nms_preds:
            # Rescale box coords to img size
            det[:,:4] = scale_coords( yoloImg.shape[2:], det[:,:4], img.shape).round()
            for x1,y1,x2,y2, conf, objclass in reversed(det):
                w,h = int(x2-x1), int(y2-y1)
                x1,y1 = int(x1), int(y1)
                results.append( ((x1,y1), (w,h), int(conf), int(objclass)))
        return results


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


