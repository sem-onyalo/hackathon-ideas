import cv2 as cv
import math
import numpy as np
import pytesseract

from PIL import Image
from core import Constants
from core import ModelManager, VideoManager

class BrandOcr:
    _appName = "brandSpotterOcr"

    isDebug = False
    showVideo = True
    writeVideo = False
    rectColor = (0, 0, 255)
    rectThickness = 5

    def __init__(self, args):
        modelObject = {
            'type': Constants.OCR,
            'obj': ModelManager.ModelManager.models[Constants.MODEL_INDEX_TEXT_DETECTOR]
        }

        self.isDebug = args.debug
        self.showVideo = not args.hide
        self.videoManager = VideoManager.VideoManager(Constants.WINDOW_NAME, modelObject, args)

    def labelDetections(self, detections):
        for detection in detections:
            for line in detection['lines']:
                self.videoManager.addLine(line.pt1.toTuple(), line.pt2.toTuple(), self.rectColor, self.rectThickness)
            self.videoManager.addLabel(detection['text'], detection['lines'][1].pt1.x, detection['lines'][1].pt1.y)

    def run(self):
        if self.isDebug:
            if not self.videoManager.targetPath:
                raise RuntimeError('Please define an --out path in debug mode')

            self.videoManager.runOcrDetection()

            detections = self.videoManager.findTextDetected()

            self.labelDetections(detections)

            self.videoManager.writeFrameAsImage()
        else:
            self.videoManager.readNewFrame()
            while not self.videoManager.img is None:
                cmd = self.videoManager.getKeyPress()
                if cmd == 27: # ESC
                    break

                self.videoManager.runOcrDetection()

                detections = self.videoManager.findTextDetected()

                self.labelDetections(detections)

                if self.showVideo:
                    self.videoManager.showImage()

                self.videoManager.writeFrame()

                self.videoManager.readNewFrame()

            self.videoManager.shutdown()
