import cv2
from . utils import *
from scipy.spatial import distance
from collections import OrderedDict
import numpy as np

from . bbox import BBox
from . bboxTracker import BBoxTracker

METAKEY_TRACKER = "objTracker_TRACKER"


class ObjectTracker:
    class Tracker:
        def __init__(self, trackerType: str, image: np.ndarray, bbox: BBox):
            self.lastSeen: BBox = bbox
            self.tracker: cv2.Tracker = ObjectTracker.createTrackerByType(trackerType)

            imgY, imgX = image.shape[:2]
            self.tracker.init(image, bbox.asX1Y1WH(imgX, imgY))

        def update(self, img):
            success, coords = self.tracker.update(img)
            if success:
                imgY, imgX = img.shape[:2]
                self.lastSeen = BBox.fromX1Y1WH(*coords, imgX, imgY)
            return success, self.lastSeen

    def __init__(self, trackerType: str = "KCF", distThresh=0.1):
        self._bboxTracker: BBoxTracker = BBoxTracker(distThresh=distThresh)
        self._trackerType = trackerType

    def getTrackedObjects(self) -> dict[int, BBoxTracker.Tracker]:
        ''' Returns a dictionary of the objects being tracked '''
        return self._bboxTracker.getTrackedObjects()

    def updateBox(self, key: int, bbox: BBox = None, metadata: dict = None):
        ''' Updates a tracked object with a new bbox or metadata '''
        return self._bboxTracker.updateBox(key=key, bbox=bbox, metadata=metadata)

    def removeBox(self, key: int):
        ''' Remove a tracked object identified by key '''
        self._bboxTracker.removeBox(key)

    def update(self, image: np.ndarray, detections: list[BBox] = None, **bboxTrackerKwargs) -> tuple[dict[int, Tracker], set[int], set[int], list[int]]:
        ''' Updates the object trackers with a new image, and optionally detections

        Parameters:
        image (np.ndarray) - image to apply tracking to
        detections (list, optional) - List of detected BBox to pass into tracker
        bboxTrackerKwargs - extra kwargs to pass to BBoxTracker, such as metadata and metadataComp

        Returns:
        ( trackedItems (dict[int,Tracker]), newKeys (set[int]), lostKeys (set[int]), matchedKeys (list[int]) )
            where trackedItems is a dictionary of all tracked items, newKeys is the set of keys that are new this
            updates, lostKeys is the set of keys that were lost this update, and matchedKeys is an list of keys
            index-matched with the passed in detections list
        '''
        trackedObjs: dict[int, ObjectTracker.Tracker] = {}
        lostKeys: set[int] = set()
        newKeys: set[int] = set()
        matchedKeys: list[int] = []

        if detections is None:
            # Run image tracking only
            trackedObjs = self._bboxTracker.getTrackedObjects()
            for key, obj in trackedObjs.items():
                tracker: ObjectTracker.Tracker = obj.metadata[METAKEY_TRACKER]
                success, bbox = tracker.update(image)
                if success:
                    self._bboxTracker.updateBox(key, bbox=bbox)
                    matchedKeys.append(key)
                else:
                    lostKeys.add(key)
        else:
            trackedObjs, newKeys, lostKeys, matchedKeys = self._bboxTracker.update(detections, **bboxTrackerKwargs)

            for key, obj in trackedObjs.items():
                if key not in lostKeys:
                    # Create/Update tracker with detection
                    tracker = ObjectTracker.Tracker(trackerType=self._trackerType, image=image, bbox=obj.bbox)
                    obj.metadata[METAKEY_TRACKER] = tracker
                    self._bboxTracker.updateBox(key, metadata=obj.metadata)

        return (trackedObjs, newKeys, lostKeys, matchedKeys)

    @staticmethod
    def createTrackerByType(trackerType: str) -> cv2.Tracker:
        trackerType = trackerType.upper()
        if trackerType == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        elif trackerType == 'MIL':
            tracker = cv2.TrackerMIL_create()
        elif trackerType == 'KCF':
            tracker = cv2.TrackerKCF_create()
        elif trackerType == 'TLD':
            tracker = cv2.TrackerTLD_create()
        elif trackerType == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        elif trackerType == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        elif trackerType == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()
        elif trackerType == "CSRT":
            tracker = cv2.TrackerCSRT_create()
        else:
            tracker = None
        return tracker
