from __future__ import annotations
import math
import numpy as np


class BBox:
    EPSILON_DIST = 0.01
    EPSILON_SIZE = .9

    def __init__(self, bbox):
        # Ensure positive w,h
        x1, y1, w, h = bbox
        if w < 0:
            x1 += w
            w = -w
        if h < 0:
            y1 += h
            h = -h
        self.bbox = np.array((x1, y1, w, h), dtype=np.float)  # Relative, x1y1, w,h

    def __eq__(self, other):
        return self.similar(other)

    def __repr__(self):
        return f"TrackerTools.BBox: {self.bbox}"

    def copy(self):
        return BBox(self.bbox)

    def dist(self, other):
        pt1 = self.bbox[:2]
        pt2 = other.bbox[:2]
        return math.dist(pt1, pt2)

    def similar(self, other, distEpsilon=EPSILON_DIST, sizeEpsilon=EPSILON_SIZE):
        if self.bbox is other.bbox:
            return True
        absAreaDiff = abs(self.area - other.area)
        areaRatio = min(self.area, other.area) / max(self.area, other.area)
        if absAreaDiff > distEpsilon and areaRatio > sizeEpsilon:
            return False
        return self.dist(other) < distEpsilon

    @property
    def area(self) -> float:
        return self.bbox[2] * self.bbox[3]

    @staticmethod
    def fromX1Y1X2Y2(x1, y1, x2, y2, imgX, imgY) -> BBox:
        return BBox((x1/imgX, y1/imgY, (x2-x1)/imgX, (y2-y1)/imgY))

    @staticmethod
    def fromX1Y1WH(x1, y1, w, h, imgX, imgY) -> BBox:
        return BBox((x1/imgX, y1/imgY, w/imgX, h/imgY))

    @staticmethod
    def fromYolo(cX, cY, w, h) -> BBox:
        return BBox((cX - w/2, cY - h/2, w, h))

    @staticmethod
    def fromRX1Y1WH(cX, cY, w, h) -> BBox:
        return BBox((cX, cY, w, h))

    def asX1Y1X2Y2(self, imgW, imgH) -> tuple[int, int, int, int]:
        x1, y1, w, h = self.asX1Y1WH(imgW, imgH)
        return((x1, y1, x1+w, y1+h))

    def asX1Y1WH(self, imgW, imgH) -> tuple[int, int, int, int]:
        x1 = round(self.bbox[0] * imgW)
        y1 = round(self.bbox[1] * imgH)
        w = round(self.bbox[2]*imgW)
        h = round(self.bbox[3]*imgH)
        return (x1, y1, w, h)

    def asYolo(self) -> tuple[float, float, float, float]:
        rX, rY, rW, rH = self.bbox
        cX, cY = rX + rW/2, rY + rH/2
        return (cX, cY, rW, rH)

    def asRX1Y1WH(self) -> tuple[float, float, float, float]:
        return tuple(self.bbox)
