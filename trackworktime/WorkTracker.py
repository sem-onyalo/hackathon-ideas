import json
import time
from core import Point, Rectangle

class WorkTracker:
    _appName = "workTracker"

    paused = False
    settings = None
    rectThickness = 2
    videoManager = None
    workPosition = None
    classToDetect = 'person'
    colorClick = (0, 0, 255)
    workPositionThreshold = 0
    colorWorking = (0, 255, 0)
    colorNotWorking = (0, 0, 255)
    videoManagerInitTimeoutSec = 5
    def __init__(self, videoManager):
        self.videoManager = videoManager
        self.videoManager.doFlipFrame = False
        self.videoManager.setMouseCallback(self.mouseCallback)
        self.loadSettings()

    def getSettings(self, name): # TODO: move to parent class
        with open('app.settings.json', 'r') as fh:
            settings = json.loads(fh.read())
            if name in settings:
                return settings[name]

        raise RuntimeError(f'Name {name} not in settings file')

    def loadSettings(self):
        self.settings = self.getSettings(self._appName)
        workPositionPts = list(map(int, self.settings['workPosition'].split(',')))
        self.workPosition = Rectangle.Rectangle(Point.Point(workPositionPts[0], workPositionPts[1]), Point.Point(workPositionPts[2], workPositionPts[3]))
        self.workPositionThreshold = self.settings['workPositionThreshold']

    def getBestDetectionHandler(self):
        handler = lambda cols, rows, detection, className, score : self.labelWorkingPositions(cols, rows, detection, className, score)
        return handler

    def labelWorkingPositions(self, cols, rows, detection, className, score):
        labelSize = 1
        self.videoManager.addRectangle(self.workPosition.pt1.toTuple(), self.workPosition.pt2.toTuple(), self.colorWorking, self.rectThickness)
        if not detection is None:
            isWithinPt1 = detection.pt1.x > (self.workPosition.pt1.x - self.workPositionThreshold) and detection.pt1.y > (self.workPosition.pt1.y - self.workPositionThreshold)
            isWithinPt2 = detection.pt2.x < (self.workPosition.pt2.x + self.workPositionThreshold) and detection.pt2.y < (self.workPosition.pt2.y + self.workPositionThreshold)
            if isWithinPt1 and isWithinPt2:
                self.videoManager.addLabel('WORKING', self.workPosition.pt1.x, self.workPosition.pt1.y, labelSize)
            else:
                self.videoManager.addRectangle(detection.pt1.toTuple(), detection.pt2.toTuple(), self.colorNotWorking, self.rectThickness)
                self.videoManager.addLabel('NOT WORKING', detection.pt1.x, detection.pt1.y, labelSize)

    def run(self):
        self.videoManager.readNewFrame()
        while not self.videoManager.cap is None:
            cmd = self.videoManager.getKeyPress()
            if cmd == 27: # ESC
                break
            elif cmd == 80 or cmd == 112: # P/p
                self.paused = not self.paused

            if self.paused:
                self.videoManager.showImage()
                continue

            self.videoManager.runDetection()
            self.videoManager.findBestDetection(self.classToDetect, self.getBestDetectionHandler())

            self.videoManager.showImage()
            self.videoManager.writeFrame()
            self.videoManager.readNewFrame()

        self.videoManager.shutdown()

    def mouseCallback(self, event, x, y, flags, param):
        if self.videoManager.isLeftButtonClick(event):
            self.videoManager.addCircle(x, y, self.colorClick)
            print(f'Mouse button click at ({x},{y})')
