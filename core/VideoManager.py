import cv2 as cv
import time
from . import Point, Rectangle, Utils

class VideoManager:
    imgMargin = 60
    img = None
    netModel = None
    windowName = ""
    detections = None
    scoreThreshold = None
    xLeftPos = None
    xRightPos = None
    yTopPos = None
    yBottomPos = None
    videoSource = None
    videoTarget = None
    doFlipFrame = True
    videoWriter = None

    def __init__(self, windowName, videoSource, videoTarget, frameWidth, frameHeight, netModel, scoreThreshold):
        self.netModel = netModel
        self.windowName = windowName
        self.frameWidth = frameWidth
        self.frameHeight = frameHeight
        self.videoSource = videoSource
        self.videoTarget = videoTarget
        self.scoreThreshold = scoreThreshold
        cv.namedWindow(self.windowName, cv.WINDOW_NORMAL)
        self.initVideoIO()

    def getImage(self):
        return self.img

    def getXCoordDetectionDiff(self):
        return self.xRightPos - self.xLeftPos if self.xRightPos != None and self.xLeftPos != None else None

    def getYCoordDetectionDiff(self):
        return self.yBottomPos - self.yTopPos if self.yBottomPos != None and self.yTopPos != None else None

    def getDefaultFont(self):
        return cv.FONT_HERSHEY_SIMPLEX

    def getKeyPress(self):
        return cv.waitKey(1)

    def getTextSize(self, text, font, scale, thickness):
        return cv.getTextSize(text, font, scale, thickness)[0]

    def showImage(self):
        cv.imshow(self.windowName, self.img)

    def addText(self, text, pt, font, scale, color, thickness):
        cv.putText(self.img, text, pt, font, scale, color, thickness, cv.LINE_AA)

    def addRectangle(self, pt1, pt2, color, thickness, isFilled=False):
        if isFilled:
            thickness = cv.FILLED
        cv.rectangle(self.img, pt1, pt2, color, thickness, cv.LINE_AA)

    def addLine(self, pt1, pt2, color, thickness):
        cv.line(self.img, pt1, pt2, color, thickness)

    def addLabel(self, label, xLeft, yTop):
        labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        yTopText = max(yTop, labelSize[1])
        cv.rectangle(self.img, (xLeft, yTopText - labelSize[1]), (xLeft + labelSize[0], yTopText + baseLine),
            (255, 255, 255), cv.FILLED)
        cv.putText(self.img, label, (xLeft, yTopText), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    def shutdown(self):
        cv.destroyAllWindows()
        if not self.videoWriter is None:
            self.videoWriter.release()

    def decode_fourcc(self, v):
        v = int(v)
        return "".join([chr((v >> 8 * i) & 0xFF) for i in range(4)])

    def initVideoIO(self):
        # common logic
        self.cvNet = cv.dnn.readNetFromTensorflow(self.netModel['modelPath'], self.netModel['configPath'])
        self.cap = cv.VideoCapture(self.videoSource)
        # self.cap = cv.VideoCapture(cv.CAP_DSHOW + self.videoSource)
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError('Warning: unable to open video source: ', self.videoSource)
        ret, _ = self.cap.read()
        if not ret:
            raise RuntimeError('Error: could not read video frame. Try changing resolution.')
        # self.img = img

        # real-time or video file detection
        isVideoSourceInt = Utils.isInteger(self.videoSource)

        if isVideoSourceInt: # real-time detection
            self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.frameWidth)
            self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.frameHeight)
        else: # video file detection
            ex = int(self.cap.get(cv.CAP_PROP_FOURCC))
            fs = int(self.cap.get(cv.CAP_PROP_FPS))
            sz = (int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
            self.videoWriter = cv.VideoWriter(self.videoTarget, ex, fs, sz, True)
            if self.videoWriter is None or not self.videoWriter.isOpened():
                raise RuntimeError('Error: unable to open video target path')

    def readNewFrame(self):
        _, img = self.cap.read()
        self.img = cv.flip(img, 1) if self.doFlipFrame else img

    def writeFrame(self):
        self.videoWriter.write(self.img)

    def runDetection(self):
        self.cvNet.setInput(cv.dnn.blobFromImage(self.img, 1.0/127.5, (300, 300), (127.5, 127.5, 127.5), swapRB=True, crop=False))
        self.detections = self.cvNet.forward()

    def findDetections(self, classNames, objectDetectedHandler=None):
        rectangles = []
        self.xLeftPos = None
        self.xRightPos = None
        self.yTopPos = None
        self.yBottomPos = None
        rows = self.img.shape[0]
        cols = self.img.shape[1]
        for detection in self.detections[0,0,:,:]:
            score = float(detection[2])
            class_id = int(detection[1])
            if score > self.scoreThreshold and self.netModel['classNames'][class_id] in classNames:
                self.xLeftPos = int(detection[3] * cols) # marginLeft
                self.yTopPos = int(detection[4] * rows) # marginTop
                self.xRightPos = int(detection[5] * cols)
                self.yBottomPos = int(detection[6] * rows)
                if objectDetectedHandler != None:
                    objectDetectedHandler(cols, rows, self.xLeftPos, self.yTopPos, self.xRightPos, self.yBottomPos, self.netModel['classNames'][class_id])
                rectangles.append(Rectangle.Rectangle(Point.Point(self.xLeftPos, self.yTopPos), Point.Point(self.xRightPos, self.yBottomPos)))
        return rectangles
