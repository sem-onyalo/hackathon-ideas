import cv2 as cv
import math
import numpy as np
import pytesseract

from PIL import Image
from core import Constants
from core import ModelManager, VideoManager, App
from fuzzywuzzy import process

class BrandOcr(App.App):
    _appName = "brandSpotter"

    brands = []
    isDebug = False
    showVideo = True
    writeVideo = False
    brandMatchRectColor = (0, 255, 0)
    brandNoMatchRectColor = (0, 0, 255)
    brandMatchRectThickness = 5
    brandLabelSize = 2.3
    brandConfidenceLabelSize = 0.7
    brandLabelThickness = 2
    brandLabelColor = (255,255,255)
    brandLabelBackgroundColor = (43,144,253)
    

    def __init__(self, args):
        modelObject = {
            'type': Constants.OCR,
            'obj': ModelManager.ModelManager.models[Constants.MODEL_INDEX_TEXT_DETECTOR]
        }

        self.isDebug = args.debug
        self.showVideo = not args.hide
        self.fuzzyMatchThreshold = args.fzm_threshold
        self.videoManager = VideoManager.VideoManager(Constants.WINDOW_NAME, modelObject, args)
        self.loadSettings()

    def loadSettings(self):
        settings = self.getSettings(self._appName)
        self.brands = settings['brands']

    def getFuzzyMatch(self, detections, threshold):
        bestMatch = None
        bestMatchScore = 0
        originalDetection = None
        for detection in detections:
            fuzzyMatch = process.extractOne(detection['text'], self.brands)
            if fuzzyMatch != None and (float(fuzzyMatch[1] / 100) >= threshold) and (bestMatch == None or bestMatchScore < fuzzyMatch[1]):
                bestMatch = fuzzyMatch[0]
                bestMatchScore = fuzzyMatch[1]
                originalDetection = detection['text']

        return (bestMatch, bestMatchScore, originalDetection)

    def labelBrandMatch(self, detections, fuzzyMatch, showAll=False):
        rectColor = self.brandMatchRectColor if fuzzyMatch[0] != None else self.brandNoMatchRectColor
        for detection in detections:
            if fuzzyMatch[0] == None or fuzzyMatch[2] == detection['text']:
                for line in detection['lines']:
                    self.videoManager.addLine(line.pt1.toTuple(), line.pt2.toTuple(), rectColor, self.brandMatchRectThickness)

            if fuzzyMatch[2] == detection['text'] or showAll:
                self.videoManager.addLabel(detection['text'], detection['lines'][1].pt1.x, detection['lines'][1].pt1.y)

        if fuzzyMatch[0] != None:
            text = fuzzyMatch[0]
            conf = f'confidence: {fuzzyMatch[1]}%'
            self.videoManager.addTextWithBackground((text, conf), 
                                                    (self.brandLabelSize, self.brandConfidenceLabelSize), 
                                                    self.brandLabelColor, 
                                                    self.brandLabelThickness, 
                                                    self.brandLabelBackgroundColor, 
                                                    'top-right')

    def run(self):
        if self.isDebug:
            if not self.videoManager.targetPath:
                raise RuntimeError('Please define an --out path in debug mode')

            self.videoManager.runOcrDetection()

            detections = self.videoManager.findTextDetected()

            fuzzyMatch = self.getFuzzyMatch(detections, self.fuzzyMatchThreshold)

            self.labelBrandMatch(detections, fuzzyMatch, True)

            self.videoManager.writeFrameAsImage()
        else:
            self.videoManager.readNewFrame()
            while not self.videoManager.img is None:
                cmd = self.videoManager.getKeyPress()
                if cmd == 27: # ESC
                    break

                self.videoManager.runOcrDetection()

                detections = self.videoManager.findTextDetected()

                fuzzyMatch = self.getFuzzyMatch(detections, self.fuzzyMatchThreshold)

                self.labelBrandMatch(detections, fuzzyMatch)

                if self.showVideo:
                    self.videoManager.showImage()

                self.videoManager.writeFrame()

                self.videoManager.readNewFrame()

            self.videoManager.shutdown()
