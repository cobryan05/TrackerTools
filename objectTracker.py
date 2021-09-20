import cv2
from . utils import *
from scipy.spatial import distance
from collections import OrderedDict
import numpy as np

from . bbox import BBox

class ObjectTracker:
    class Tracker:
        def __init__(self):
            self.tracker : cv2.Tracker = None
            self.lastSeen : BBox = None
            self.lostCount = 0
            self.metadata = None

        def update(self, img):
            if self.tracker:
                return self.tracker.update(img)
            else:
                return None

    def __init__(self, trackerType:str = "KCF" ):
        self._trackers : OrderedDict[int, ObjectTracker.Tracker] = OrderedDict()
        self._trackerType = trackerType
        self._lastTrackerKey = 0
        self._image : np.ndarray = None
        self._distThreshold = 0.1
        self._missingFrames = 5



    """ Adds a bounding box to track, in rxywh format """
    def addObject(self, bbox: BBox, metadata = None):
        if self._image is None:
            raise Exception("Set image first")


        newTracker = ObjectTracker.Tracker()
        newTracker.tracker = ObjectTracker.createTrackerByType( self._trackerType )
        newTracker.lastSeen = bbox.copy()
        imgY, imgX = self._image.shape[:2]
        newTracker.tracker.init( self._image, bbox.asX1Y1WH( imgX, imgY ) )
        newTracker.metadata = metadata
        self._lastTrackerKey += 1
        self._trackers[self._lastTrackerKey] = newTracker
        print(f"addObject @ {self._lastTrackerKey} {bbox}")
        return self._lastTrackerKey



    def clear(self):
        self._trackers.clear()



    def setImage(self, image: np.ndarray):
        self._image = image.copy()



    def update(self) -> dict[int, BBox]:
        res = {}
        for key,tracker in self._trackers.items():
            result = tracker.update( self._image )
            success,bbox_coords = result
            if success:
                imgY, imgX = self._image.shape[:2]
                bbox = BBox.fromX1Y1WH( *bbox_coords, imgX, imgY )
                tracker.lastSeen = bbox
            else:
                bbox = None
            res[key] = bbox
        return res


    """ Update the tracker with YOLO detections """
    def updateDetections(self, detections: list[BBox], detectionMetadata):
        # Try to match detections with currently tracked objects
        updatedTrackedObjs, matchedKeys, lostObjIds = self.matchDetections( detections, detectionMetadata )

        # Update any already-tracked objects
        for key,tracker in updatedTrackedObjs.items():
            self._trackers[key].tracker = tracker

        # Track any new objects
        for idx,key in enumerate(matchedKeys):
            if key is None:
                self.addObject( detections[idx], detectionMetadata[idx])
            else:
                pass # Already tracked object

        # Handle lost objects:
        for id in lostObjIds:
            print(f"Lost id: {id}")
            self._trackers[id].lostCount += 1
        pass


    # TODO: Inconsistent use of detectedClasses/metadata
    def matchDetections( self, detectedObjects, detectedClasses):
        # Try to match up new detections with tracked objects

        # First gather all of the last-seen positions for tracked objects
        trackedObjectIds = []
        trackedObjectCoords = []
        for key,value in self._trackers.items():
            trackedObjectIds.append(key)
            trackedObjectCoords.append(value.lastSeen)

        # Try to match detections with currently tracked objects
        updatedTrackedObjs, matchedKeys, lostObjIds = self._matchDetections(
                                    np.array( detectedObjects ),
                                    np.array( trackedObjectCoords ),
                                    trackedObjectIds )

        return (updatedTrackedObjs, matchedKeys, lostObjIds )


    # Returns updatedTrackedObjs, matchedKeys, lostIds
    # matchedKeys indexes correspond to detectedObject indexes
    def _matchDetections( self, detectedObjects : list[BBox], trackedObjects : list[BBox], trackedIds : list[int] ):
        updatedTrackedObjs = {}
        matchedIds = [None]*len(detectedObjects)

        # If there are no currently tracked objects then all objects are new
        if len(trackedObjects) == 0:
            return ( updatedTrackedObjs, matchedIds, [] )

        # No detections, then all objects are lost
        if len(detectedObjects) == 0:
            return ( updatedTrackedObjs, matchedIds, trackedIds )


        lostObjectIds = []

        detectedCoords = [ obj.bbox for obj in detectedObjects]
        trackedCoords = [ obj.bbox for obj in trackedObjects]
        # Adapted from
        # https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
        dist = distance.cdist( trackedCoords, detectedCoords )

        rows = dist.min(axis=1).argsort()
        cols = dist.argmin(axis=1)[rows]
        usedRows = set()
        usedCols = set()
        imgY, imgX = self._image.shape[:2]
        for (row, col) in zip(rows, cols):
            if row in usedRows or col in usedCols:
                continue

            objId = trackedIds[row]
            matchedIds[col] = objId
            bbox = detectedObjects[col]
            # Create a new tracker for this detection, return it with the same id
            newTracker = ObjectTracker.createTrackerByType( self._trackerType )
            newTracker.init( self._image, bbox.asX1Y1WH(imgX, imgY))
            updatedTrackedObjs[objId] = newTracker

            usedRows.add(row)
            usedCols.add(col)

        # compute both the row and column index we have NOT yet
        # examined
        unusedRows = set(range(0, dist.shape[0])).difference(usedRows)
        unusedCols = set(range(0, dist.shape[1])).difference(usedCols)

        # If we were tracking more than we detected then we lost objects
        if dist.shape[0] >= dist.shape[1]:
            for row in unusedRows:
                objId = trackedIds[row]
                lostObjectIds.append( objId )

        return ( updatedTrackedObjs, matchedIds, lostObjectIds )



    @staticmethod
    def createTrackerByType( trackerType:str ) -> cv2.Tracker:
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
