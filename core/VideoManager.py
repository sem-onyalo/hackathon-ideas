import os
import cv2 as cv
import time
import math
import numpy as np
import pytesseract

from PIL import Image
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
    sourcePath = None
    targetPath = None
    doFlipFrame = False
    videoWriter = None
    detectorFeaturesOut = []

    def __init__(self, windowName, modelObject, args):
        self.windowName = windowName
        self.sourcePath = args.sourcePath
        self.targetPath = args.out
        self.scoreThreshold = args.score_threshold
        self.frameWidth, self.frameHeight = self.parseResolution(args.res)
        self.preprocWidth, self.preprocHeight = self.parseResolution(args.res_preproc)
        self.detectorFeaturesOut.append("feature_fusion/Conv_7/Sigmoid")
        self.detectorFeaturesOut.append("feature_fusion/concat_3")

        cv.namedWindow(self.windowName, cv.WINDOW_NORMAL)

        self.initVideoIO()
        self.initDnnModel(modelObject)

    # ----------------------------------------------------------------------------------------------------
    #     Setup and Teardown Methods
    # ----------------------------------------------------------------------------------------------------
    
    def initVideoIO(self):
        # common logic
        self.cap = cv.VideoCapture(self.sourcePath)
        # self.cap = cv.VideoCapture(cv.CAP_DSHOW + self.sourcePath)
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError('Warning: unable to open video source: ', self.sourcePath)
        # ret, _ = self.cap.read()
        ret, self.img = self.cap.read()
        if not ret:
            raise RuntimeError('Error: could not read video frame. Try changing resolution.')

        # real-time or video file detection
        isVideoSourceInt = Utils.isInteger(self.sourcePath)

        if isVideoSourceInt: # real-time detection
            self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.frameWidth)
            self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.frameHeight)
        else: # video file detection
            ex = int(self.cap.get(cv.CAP_PROP_FOURCC))
            fs = int(self.cap.get(cv.CAP_PROP_FPS))
            sz = (int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
            if self.isVideoTargetPath(self.targetPath):
                self.videoWriter = cv.VideoWriter(self.targetPath, ex, fs, sz, True)
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

    def writeFrameAsImage(self):
        cv.imwrite(self.targetPath, self.img)

    def showImage(self):
        cv.imshow(self.windowName, self.img)

    # ----------------------------------------------------------------------------------------------------
    #     Misc Helper Methods
    # ----------------------------------------------------------------------------------------------------
    
    def isLeftButtonClick(self, event):
        return event == cv.EVENT_LBUTTONDOWN

    def isVideoTargetPath(self, targetPath):
        if not targetPath:
            return False
        else:
            _, ext = os.path.splitext(targetPath)
            return ext in ('.mp4')

    def decode_fourcc(self, v):
        v = int(v)
        return "".join([chr((v >> 8 * i) & 0xFF) for i in range(4)])

    def parseResolution(self, resolution):
        resSplit = list(map(int, resolution.split(',')))
        width = resSplit[0]
        height = resSplit[1]
        return width, height

    def fourPointsTransform(self, frame, vertices):
        vertices = np.asarray(vertices)
        outputSize = (100, 32)
        targetVertices = np.array([
            [0, outputSize[1] - 1],
            [0, 0],
            [outputSize[0] - 1, 0],
            [outputSize[0] - 1, outputSize[1] - 1]], dtype="float32")

        rotationMatrix = cv.getPerspectiveTransform(vertices, targetVertices)
        result = cv.warpPerspective(frame, rotationMatrix, outputSize)
        return result

    def decodeText(self, scores):
        text = ""
        alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
        for i in range(scores.shape[0]):
            c = np.argmax(scores[i][0])
            if c != 0:
                text += alphabet[c - 1]
            else:
                text += '-'

        # adjacent same letters as well as background text must be removed to get the final output
        char_list = []
        for i in range(len(text)):
            if text[i] != '-' and (not (i > 0 and text[i] == text[i - 1])):
                char_list.append(text[i])
        return ''.join(char_list)

    def decodeBoundingBoxes(self, scores, geometry, scoreThresh):
        detections = []
        confidences = []

        ############ CHECK DIMENSIONS AND SHAPES OF geometry AND scores ############
        assert len(scores.shape) == 4, "Incorrect dimensions of scores"
        assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
        assert scores.shape[0] == 1, "Invalid dimensions of scores"
        assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
        assert scores.shape[1] == 1, "Invalid dimensions of scores"
        assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
        assert scores.shape[2] == geometry.shape[2], "Invalid dimensions of scores and geometry"
        assert scores.shape[3] == geometry.shape[3], "Invalid dimensions of scores and geometry"
        height = scores.shape[2]
        width = scores.shape[3]
        for y in range(0, height):

            # Extract data from scores
            scoresData = scores[0][0][y]
            x0_data = geometry[0][0][y]
            x1_data = geometry[0][1][y]
            x2_data = geometry[0][2][y]
            x3_data = geometry[0][3][y]
            anglesData = geometry[0][4][y]
            for x in range(0, width):
                score = scoresData[x]

                # If score is lower than threshold score, move to next x
                if (score < scoreThresh):
                    continue

                # Calculate offset
                offsetX = x * 4.0
                offsetY = y * 4.0
                angle = anglesData[x]

                # Calculate cos and sin of angle
                cosA = math.cos(angle)
                sinA = math.sin(angle)
                h = x0_data[x] + x2_data[x]
                w = x1_data[x] + x3_data[x]

                # Calculate offset
                offset = ([offsetX + cosA * x1_data[x] + sinA * x2_data[x], offsetY - sinA * x1_data[x] + cosA * x2_data[x]])

                # Find points for rectangle
                p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
                p3 = (-cosA * w + offset[0], sinA * w + offset[1])
                center = (0.5 * (p1[0] + p3[0]), 0.5 * (p1[1] + p3[1]))
                detections.append((center, (w, h), -1 * angle * 180.0 / math.pi))
                confidences.append(float(score))

        # Return detections and confidences
        return [detections, confidences]

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
        self.detections = self.cvNet.forward(self.detectorFeaturesOut)

    def findTextDetected(self):
        textDetected = []

        scores = self.detections[0]
        geometry = self.detections[1]
        [boxes, confidences] = self.decodeBoundingBoxes(scores, geometry, self.scoreThreshold)

        rW = self.img.shape[1] / float(self.preprocWidth)
        rH = self.img.shape[0] / float(self.preprocHeight)

        # Apply NMS
        indices = cv.dnn.NMSBoxesRotated(boxes, confidences, self.scoreThreshold, self.scoreThreshold)
        for i in indices:
            # get 4 corners of the rotated rect
            vertices = cv.boxPoints(boxes[i[0]])

            # scale the bounding box coordinates based on the respective ratios
            for j in range(4):
                vertices[j][0] *= rW
                vertices[j][1] *= rH
                
            # crop image around detected text
            cropped = self.fourPointsTransform(self.img, vertices)

            # preprocess image before OCR
            preprocessed = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)
            preprocessed = cv.threshold(preprocessed, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
            preprocessed = cv.medianBlur(preprocessed, 3)
            # cv.imwrite('spotbrandloyalty/samples/preprocessing.jpg', preprocessed)

            # run OCR
            text = pytesseract.image_to_string(preprocessed, config='-l eng --oem 1 --psm 6')

            # TODO: clean text before appending
            textDetected.append(text)

            for j in range(4):
                p1 = (vertices[j][0], vertices[j][1])
                p2 = (vertices[(j + 1) % 4][0], vertices[(j + 1) % 4][1])
                cv.line(self.img, p1, p2, (0, 255, 0), 5)

            cv.putText(self.img, text, (int(vertices[1][0]), int(vertices[1][1])), cv.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 0))
            
        return textDetected