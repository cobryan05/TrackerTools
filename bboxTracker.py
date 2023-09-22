''' Assigns IDs to bounding boxes and attempts to match them between updates'''
import copy
from collections import OrderedDict
from collections.abc import Callable
from scipy.spatial import distance
import numpy as np

from . bbox import BBox


class BBoxTracker:

    class Tracker:
        def __init__(self, bbox: BBox = None, key: int = None, metadata: dict = {}):
            self.bbox: BBox = bbox
            self.key: int = key
            self.metadata: dict = metadata

    def __init__(self, distThresh: float = 0.1):
        ''' Constructor for BBox Tracker

        Parameters:
        distThresh (float): maximum distance for two bboxes to be compared as the same '''
        self._trackedObjs: OrderedDict[int, BBoxTracker.Tracker] = OrderedDict()
        self._lastKey: int = 0
        self._distThreshold: float = distThresh
        self._missingFrames: int = 5

    def addNewBox(self, bbox: BBox, metadata: dict = None) -> int:
        ''' Adds a new bounding box to be tracked

        Returns the ID of the newly tracked box '''
        self._lastKey += 1
        self._trackedObjs[self._lastKey] = BBoxTracker.Tracker(bbox=bbox, key=self._lastKey, metadata=metadata)
        return self._lastKey

    def removeBox(self, key: int):
        ''' Remove a tracked object identified by key '''
        self._trackedObjs.pop(key, None)

    def updateBox(self, key: int, bbox: BBox = None, metadata: dict = None, mergeMetadata: bool = False):
        ''' Updates a tracked object with a new bbox or metadata '''
        obj = self._trackedObjs.get(key, None)
        if obj:
            if bbox is not None:
                obj.bbox = copy.copy(bbox)
            if metadata is not None:
                if mergeMetadata:
                    obj.metadata = {**metadata, **obj.metadata}
                else:
                    obj.metadata = metadata.copy()

    def getTrackedObjects(self) -> dict[int, Tracker]:
        ''' Returns a dictionary of the objects being tracked '''
        return self._trackedObjs.copy()

    def clear(self):
        ''' Clears all tracked items '''
        self._trackedObjs.clear()

    def update(self, detections: list[BBox], metadata: list[dict] = None,
               metadataComp: Callable[[dict, dict], float] = None, mergeMetadata=False) -> tuple[dict[int, Tracker], set[int], set[int], list[int]]:
        ''' Updates tracker with new set of bboxes

        Attempts to match tracked boxes with new boxes

        Parameters:
        detections (list[BBox]) list of detected bounding boxes
        metadata (list[dict], optional) list of metadata about bboxes, index-matched to detections
        metadataComp (Callable[[dict, dict], float], optional) function that returns confidence, 0.0-1.0, that
                     two metadata dictionaries describe the same object. May be called while matching boxes.
        mergeMetadata (bool, optional) if True then the metadata dictionary will be merged with the current one, otherwise replaces it
        Returns:
        (allTrackedItems (dict), newItems (set), lostItems (set), detectedKeys (list)) where
         'allTrackedItems' is a dictionary with the tracked items, and newItems and lostItems are sets
        of new and lost item keys, and matchedKeys contains the matched keys index-matched to the input detections'''

        trackedRes: dict[int, BBoxTracker.Tracker] = {}
        lostKeySet: set[int] = set()
        newKeySet: set[int] = set()

        trackedKeys = set(self._trackedObjs.keys())
        matchedKeys = self._matchDetections(detections, metadata=metadata, metadataComp=metadataComp)
        detectedKeys = []

        for idx, (key, bbox) in enumerate(zip(matchedKeys, detections)):
            data = metadata[idx] if metadata and idx < len(metadata) else None
            if key is None:
                # This is a newly tracked object
                newKey = self.addNewBox(bbox, metadata=data)
                trackedRes[newKey] = copy.copy(self._trackedObjs[newKey])
                newKeySet.add(newKey)
                detectedKeys.append(newKey)
            else:
                # This object was already tracked
                self.updateBox(key, bbox=bbox, metadata=data, mergeMetadata=mergeMetadata)
                trackedRes[key] = copy.copy(self._trackedObjs[key])
                trackedKeys.remove(key)
                detectedKeys.append(key)

        # Any keys remaining in trackedKeys are lost
        for key in trackedKeys:
            trackedRes[key] = copy.copy(self._trackedObjs[key])
            lostKeySet.add(key)

        return trackedRes, newKeySet, lostKeySet, detectedKeys

    def _matchDetections(self, detections: list[BBox], metadata: list[dict] = None,
                         metadataComp: Callable[[tuple[BBox, dict], tuple[BBox, dict]], float] = None) -> list[int]:
        ''' Match passed in bounding boxes with nearest tracked box

        Returns a list keys with indexes matching the passed in detections list '''

        # If metadata wasn't passed in then no use using the comparison function
        if metadata is None or len(metadata) < len(detections):
            metadataComp = None

        # Gather currently tracked bboxes
        trackedBoxes = [tracker.bbox for tracker in self._trackedObjs.values()]
        trackedIds = [tracker.key for tracker in self._trackedObjs.values()]

        matchedIds: list[int] = [None]*len(detections)

        # If there were no previous boxes, or are no current boxes, then no ids are matched
        if len(trackedBoxes) == 0 or len(detections) == 0:
            return matchedIds

        trackedCoords = [bbox.bbox[:2] for bbox in trackedBoxes]
        detectedCoords = [bbox.bbox[:2] for bbox in detections]

        # Adapted from
        # https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
        dist = distance.cdist(trackedCoords, detectedCoords)
        dist[dist > self._distThreshold] = 1  # Set anything further apart than threshold to 1

        # If there is a metadata comparison function then apply it before continuing
        if metadataComp:
            # Only consider pairs that are within the tracking distance threshold
            nearbyIndexes = np.where(dist < self._distThreshold)
            nearbyPairs = list(zip(nearbyIndexes[0], nearbyIndexes[1]))

            for trackedIdx, detectedIdx in nearbyPairs:
                trackedKey = trackedIds[trackedIdx]
                trackedObj = self._trackedObjs[trackedKey]
                leftMeta = trackedObj.metadata
                leftBbox = trackedObj.bbox
                rightMeta = metadata[detectedIdx]
                rightBbox = detections[detectedIdx]
                sameConf = metadataComp((leftBbox, leftMeta), (rightBbox, rightMeta))

                # Adjust distance based on confidence
                multiplier = (1.5 - sameConf)
                dist[trackedIdx][detectedIdx] *= pow(multiplier, 2)

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
