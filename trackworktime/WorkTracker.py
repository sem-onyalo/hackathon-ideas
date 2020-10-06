import json
import time
from core import Constants, Point, Rectangle
from core import ModelManager, VideoManager

class WorkTracker:
    _appName = "workTracker"

    paused = False
    settings = None
    startTime = None
    isWorking = None
    workingTime = None
    videoManager = None
    workPosition = None
    rectangleThickness = 2
    totalWorkingTime = None
    classToDetect = 'person'
    colorClick = (0, 0, 255)
    workPositionThreshold = 0
    colorWorking = (0, 255, 0)
    colorNotWorking = (0, 0, 255)
    videoManagerInitTimeoutSec = 5

    def __init__(self, args):
        res = list(map(int, args.res.split(',')))
        model = ModelManager.ModelManager.models[Constants.MODEL_INDEX_COCO]
        self.videoManager = VideoManager.VideoManager(Constants.WINDOW_NAME, args.sourcePath, args.out, res[0], res[1], model, args.score_threshold)
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
        if self.settings['workPosition'] != "":
            workPositionPts = list(map(int, self.settings['workPosition'].split(',')))
            self.workPosition = Rectangle.Rectangle(Point.Point(workPositionPts[0], workPositionPts[1]), Point.Point(workPositionPts[2], workPositionPts[3]))
        self.workPositionThreshold = self.settings['workPositionThreshold']

    def getBestDetectionHandler(self):
        handler = lambda cols, rows, detection, className, score : self.labelWorkingPositions(cols, rows, detection, className, score)
        return handler

    def labelWorkingPositions(self, cols, rows, detection, className, score):
        labelSize = 0.7
        if not self.workPosition is None:
            self.videoManager.addRectangle(self.workPosition.pt1.toTuple(), self.workPosition.pt2.toTuple(), self.colorWorking, self.rectangleThickness)
            if not detection is None:
                isWithinPt1 = detection.pt1.x > (self.workPosition.pt1.x - self.workPositionThreshold) and detection.pt1.y > (self.workPosition.pt1.y - self.workPositionThreshold)
                isWithinPt2 = detection.pt2.x < (self.workPosition.pt2.x + self.workPositionThreshold) and detection.pt2.y < (self.workPosition.pt2.y + self.workPositionThreshold)
                isWorking = isWithinPt1 and isWithinPt2

                if self.isWorking is None:
                    self.isWorking = isWorking
                    if self.isWorking:
                        self.workingTime = time.time()
                elif not self.isWorking and isWorking:
                    self.isWorking = isWorking
                    self.workingTime = time.time()
                elif self.isWorking and not isWorking:
                    self.isWorking = isWorking
                    self.totalWorkingTime += (time.time() - self.workingTime)

                if isWorking:
                    self.videoManager.addLabel('WORKING', self.workPosition.pt1.x, self.workPosition.pt1.y, labelSize)
                else:
                    self.videoManager.addRectangle(detection.pt1.toTuple(), detection.pt2.toTuple(), self.colorNotWorking, self.rectangleThickness)
                    self.videoManager.addLabel('WORK POSITION', self.workPosition.pt1.x, self.workPosition.pt1.y, labelSize)
                    self.videoManager.addLabel('NOT WORKING', detection.pt1.x, detection.pt1.y, labelSize)
            else:
                self.videoManager.addLabel('WORK POSITION', self.workPosition.pt1.x, self.workPosition.pt1.y, labelSize)
        elif not detection is None:
            self.videoManager.addRectangle(detection.pt1.toTuple(), detection.pt2.toTuple(), self.colorNotWorking, self.rectangleThickness)

    def run(self):
        self.workingTime = 0
        self.isWorking = None
        self.totalWorkingTime = 0
        self.startTime = time.time()
        self.videoManager.readNewFrame()
        while not self.videoManager.img is None:
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

        elapsedTime = time.time() - self.startTime
        print(f'Time spent working: {(self.totalWorkingTime / elapsedTime):.0%} ({self.totalWorkingTime:.2f}s out of {elapsedTime:.2f}s)')

        self.videoManager.shutdown()

    def mouseCallback(self, event, x, y, flags, param):
        if self.videoManager.isLeftButtonClick(event):
            self.videoManager.addCircle(x, y, self.colorClick)
            print(f'Mouse button click at ({x},{y})')
