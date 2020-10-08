import cv2 as cv
import time
from . import Constants, Point, Rectangle, Utils

class VideoManager:
    img = None
    imgMargin = 60
    windowName = ""
    netModel = None
    detections = None
    scoreThreshold = None
    xLeftPos = None
    xRightPos = None
    yTopPos = None
    yBottomPos = None
    videoSource = None
    videoTarget = None
    doFlipFrame = False
    videoWriter = None

    def __init__(self, windowName, modelObject, args):
        self.windowName = windowName
        self.videoSource = args.sourcePath
        self.videoTarget = args.out
        self.scoreThreshold = args.score_threshold
        self.frameWidth, self.frameHeight = self.parseResolution(args.res)
        self.preprocWidth, self.preprocHeight = self.parseResolution(args.res_preproc)

        cv.namedWindow(self.windowName, cv.WINDOW_NORMAL)

        self.initVideoIO()
        self.initDnnModel(modelObject)

    # ----------------------------------------------------------------------------------------------------
    #     Setup and Teardown Methods
    # ----------------------------------------------------------------------------------------------------
    
    def initVideoIO(self):
        # common logic
        self.cap = cv.VideoCapture(self.videoSource)
        # self.cap = cv.VideoCapture(cv.CAP_DSHOW + self.videoSource)
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError('Warning: unable to open video source: ', self.videoSource)
        ret, _ = self.cap.read()
        if not ret:
            raise RuntimeError('Error: could not read video frame. Try changing resolution.')

        # real-time or video file detection
        isVideoSourceInt = Utils.isInteger(self.videoSource)

        if isVideoSourceInt: # real-time detection
            self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.frameWidth)
            self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.frameHeight)
        else: # video file detection
            ex = int(self.cap.get(cv.CAP_PROP_FOURCC))
            fs = int(self.cap.get(cv.CAP_PROP_FPS))
            sz = (int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
            if not self.videoTarget is None:
                self.videoWriter = cv.VideoWriter(self.videoTarget, ex, fs, sz, True)
                if self.videoWriter is None or not self.videoWriter.isOpened():
                    raise RuntimeError('Error: unable to open video target path')
                
    def initDnnModel(self, modelObject):
        if modelObject['type'] == Constants.OBJECT_DETECTION:
            self.netModel = modelObject['obj']
            self.cvNet = cv.dnn.readNetFromTensorflow(self.netModel['modelPath'], self.netModel['configPath'])
        elif modelObject['type'] == Constants.OCR:
            self.netModel = modelObject['obj']
            self.cvNet = cv.dnn.readNet(self.netModel['modelPath'])
        else:
            raise RuntimeError('Error: unknown model object type:', modelObject['type'])

    def shutdown(self):
        cv.destroyAllWindows()
        if not self.videoWriter is None:
            self.videoWriter.release()

    # ----------------------------------------------------------------------------------------------------
    #     Getter and Setter Methods
    # ----------------------------------------------------------------------------------------------------
    
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
        return cv.getTextSize(text, font, scale, thickness)

    def setMouseCallback(self, callbackFunction):
        cv.setMouseCallback(self.windowName, callbackFunction)

    # ----------------------------------------------------------------------------------------------------
    #     Frame Modification Methods
    # ----------------------------------------------------------------------------------------------------
    
    def addText(self, text, pt, font, scale, color, thickness):
        cv.putText(self.img, text, pt, font, scale, color, thickness, cv.LINE_AA)

    def addRectangle(self, pt1, pt2, color, thickness, isFilled=False):
        if isFilled:
            thickness = cv.FILLED
        cv.rectangle(self.img, pt1, pt2, color, thickness, cv.LINE_AA)

    def addCircle(self, pt1, pt2, color, thickness=5):
        cv.circle(self.img, (pt1, pt2), thickness, color, -1)

    def addLine(self, pt1, pt2, color, thickness):
        cv.line(self.img, pt1, pt2, color, thickness)

    def addLabel(self, label, xLeft, yTop, size=0.5):
        labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, size, 1)
        yTopText = max(yTop, labelSize[1])
        cv.rectangle(self.img, (xLeft, yTopText - labelSize[1]), (xLeft + labelSize[0], yTopText + baseLine),
            (255, 255, 255), cv.FILLED)
        cv.putText(self.img, label, (xLeft, yTopText), cv.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0))

    # ----------------------------------------------------------------------------------------------------
    #     Frame I/O Methods
    # ----------------------------------------------------------------------------------------------------
    
    def readNewFrame(self):
        _, img = self.cap.read()
        self.img = cv.flip(img, 1) if self.doFlipFrame else img

    def writeFrame(self):
        if not self.videoWriter is None:
            self.videoWriter.write(self.img)

    def showImage(self):
        cv.imshow(self.windowName, self.img)

    # ----------------------------------------------------------------------------------------------------
    #     Misc Helper Methods
    # ----------------------------------------------------------------------------------------------------
    
    def isLeftButtonClick(self, event):
        return event == cv.EVENT_LBUTTONDOWN

    def decode_fourcc(self, v):
        v = int(v)
        return "".join([chr((v >> 8 * i) & 0xFF) for i in range(4)])

    def parseResolution(self, resolution):
        resSplit = list(map(int, resolution.split(',')))
        width = resSplit[0]
        height = resSplit[1]
        return width, height

    # ####################################################################################################
    # ----------------------------------------------------------------------------------------------------
    #     Model Inference Methods
    # ----------------------------------------------------------------------------------------------------
    # ####################################################################################################
    
    # ----------------------------------------------------------------------------------------------------
    #     Object Detection
    # ----------------------------------------------------------------------------------------------------
    
    def runObjectDetection(self):
        self.cvNet.setInput(cv.dnn.blobFromImage(self.img, 1.0/127.5, (300, 300), (127.5, 127.5, 127.5), swapRB=True, crop=False))
        self.detections = self.cvNet.forward()

    def findObjectsDetected(self, classNames, objectDetectedHandler=None):
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
                    objectDetectedHandler(cols, rows, self.xLeftPos, self.yTopPos, self.xRightPos, self.yBottomPos, self.netModel['classNames'][class_id], score)
                rectangles.append(Rectangle.Rectangle(Point.Point(self.xLeftPos, self.yTopPos), Point.Point(self.xRightPos, self.yBottomPos)))
        return rectangles

    def findBestObjectDetected(self, className, findBestDetectionHandler=None):
        currentScore = 0
        bestDetection = None
        rows = self.img.shape[0]
        cols = self.img.shape[1]
        for detection in self.detections[0,0,:,:]:
            class_id = int(detection[1])
            score = float(detection[2])
            if self.netModel['classNames'][class_id] == className and score > self.scoreThreshold:
                xLeftPt = int(detection[3] * cols)
                yTopPt = int(detection[4] * rows)
                xRightPt = int(detection[5] * cols)
                yBottomPt = int(detection[6] * rows)
                currentDetection = Rectangle.Rectangle(Point.Point(xLeftPt, yTopPt), Point.Point(xRightPt, yBottomPt))
                if (bestDetection == None or (bestDetection != None and currentScore < score)):
                    currentScore = score
                    bestDetection = currentDetection

        if findBestDetectionHandler != None:
            findBestDetectionHandler(cols, rows, bestDetection, className, currentScore)

        return bestDetection

    def findClosestObjectDetected(self, className, objectDetectedHandler=None):
        currentScore = 0
        closestDetection = None
        rows = self.img.shape[0]
        cols = self.img.shape[1]
        for detection in self.detections[0,0,:,:]:
            class_id = int(detection[1])
            score = float(detection[2])
            if self.netModel['classNames'][class_id] == className and score > self.scoreThreshold:
                xLeftPt = int(detection[3] * cols)
                yTopPt = int(detection[4] * rows)
                xRightPt = int(detection[5] * cols)
                yBottomPt = int(detection[6] * rows)
                currentDetection = Rectangle.Rectangle(Point.Point(xLeftPt, yTopPt), Point.Point(xRightPt, yBottomPt))
                if (closestDetection == None or (closestDetection != None and currentDetection.getArea() > closestDetection.getArea())):
                    currentScore = score
                    closestDetection = currentDetection

        if closestDetection != None:
            if objectDetectedHandler != None:
                objectDetectedHandler(cols, rows, closestDetection.pt1.x, closestDetection.pt1.y, closestDetection.pt2.x, closestDetection.pt2.y, className, currentScore)
            return closestDetection

    # ----------------------------------------------------------------------------------------------------
    #     OCR
    # ----------------------------------------------------------------------------------------------------
    
    def runOcrDetection(self):
        self.cvNet.setInput(cv.dnn.blobFromImage(self.img, 1.0, (self.preprocWidth, self.preprocHeight), (123.68, 116.78, 103.94), True, False))
        self.detections = self.cvNet.forward(self.outNames)
