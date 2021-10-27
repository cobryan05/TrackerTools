''' Assigns IDs to bounding boxes and attempts to match them between updates'''
import copy
from collections import OrderedDict
from scipy.spatial import distance

from . bbox import BBox


class BBoxTracker:

    class Tracker:
        def __init__(self, bbox: BBox = None, key: int = None, metadata: object = None):
            self.bbox: BBox = bbox
            self.key: int = key
            self.metadata: object = metadata

    def __init__(self):
        self._trackedObjs: OrderedDict[int, BBoxTracker.Tracker] = OrderedDict()
        self._lastKey: int = 0
        self._distThreshold: float = 0.1
        self._missingFrames: int = 5

    def addNewBox(self, bbox: BBox, metadata: object = None) -> int:
        ''' Adds a new bounding box to be tracked

        Returns the ID of the newly tracked box '''
        self._lastKey += 1
        self._trackedObjs[self._lastKey] = BBoxTracker.Tracker(bbox=bbox, key=self._lastKey, metadata=metadata)
        print(f"Tracking new object: {self._lastKey} - {bbox}")
        return self._lastKey

    def removeBox(self, key: int):
        ''' Remove a tracked object identified by key '''
        self._trackedObjs.pop(key, None)

    def updateMetadata(self, key: int, metadata: object = None):
        ''' Updates metadata for a tracked object, identified by key '''
        if key in self._trackedObjs:
            self._trackedObjs[key].metadata = metadata

    def clear(self):
        ''' Clears all tracked items '''
        self._trackedObjs.clear()

    def update(self, detections: list[BBox], metadata: list[object] = None) \
            -> tuple[dict[int, Tracker], dict[int, Tracker], dict[int, Tracker]]:
        ''' Update tracker with a new set of BBoxes

        Returns (allTrackedItems, newItems, lostItems) dictionaries '''

        trackedRes = {}
        lostRes = {}
        newRes = {}

        trackedKeys = set(self._trackedObjs.keys())
        matchedKeys = self._matchDetections(detections)

        for idx, (key, bbox) in enumerate(zip(matchedKeys, detections)):
            if key is None:
                # This is a newly tracked object
                data = metadata[idx] if metadata and idx < len(metadata) else None
                newKey = self.addNewBox(bbox, metadata=data)
                trackedRes[newKey] = copy.copy(self._trackedObjs[newKey])
                newRes[newKey] = copy.copy(self._trackedObjs[newKey])
            else:
                # This object was already tracked
                trackedRes[key] = copy.copy(self._trackedObjs[key])
                trackedKeys.remove(key)

        # Any keys remaining in trackedKeys are lost
        for key in trackedKeys:
            lostRes[key] = copy.copy(self._trackedObjs[key])
            trackedRes[key] = copy.copy(self._trackedObjs[key])

        return trackedRes, newRes, lostRes

    def _matchDetections(self, detections: list[BBox]) -> list[int]:
        ''' Match passed in bounding boxes with nearest tracked box

        Returns a list keys with indexes matching the passed in detections list '''

        # Gather currently tracked bboxes
        trackedBoxes = [tracker.bbox for tracker in self._trackedObjs.values()]
        trackedIds = [tracker.key for tracker in self._trackedObjs.values()]

        matchedIds: list[int] = [None]*len(detections)

        # If there were no previous boxes, or are no current boxes, then no ids are matched
        if len(trackedBoxes) == 0 or len(detections) == 0:
            return matchedIds

        trackedCoords = [bbox.bbox for bbox in trackedBoxes]
        detectedCoords = [bbox.bbox for bbox in detections]

        # Adapted from
        # https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
        dist = distance.cdist(trackedCoords, detectedCoords)

        rows = dist.min(axis=1).argsort()
        cols = dist.argmin(axis=1)[rows]
        usedRows = set()
        usedCols = set()
        for (row, col) in zip(rows, cols):
            if row in usedRows or col in usedCols:
                continue

            objId = trackedIds[row]
            matchedIds[col] = objId

            usedRows.add(row)
            usedCols.add(col)
        return matchedIds
