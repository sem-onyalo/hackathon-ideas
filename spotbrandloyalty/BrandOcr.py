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

    def __init__(self, args):
        modelObject = {
            'type': Constants.OCR,
            'obj': ModelManager.ModelManager.models[Constants.MODEL_INDEX_TEXT_DETECTOR]
        }

        self.isDebug = args.debug
        self.videoManager = VideoManager.VideoManager(Constants.WINDOW_NAME, modelObject, args)

    def run(self):
        if self.isDebug:
            if not self.videoManager.targetPath:
                raise RuntimeError('Please define an --out path in debug mode')

            self.videoManager.runOcrDetection()

            self.videoManager.findTextDetected()

            self.videoManager.writeFrameAsImage()
