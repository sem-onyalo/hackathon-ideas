import time

class WorkTracker:
    rectThickness = 2
    videoManager = None
    classToDetect = 'person'
    colorWorking = (0, 0, 255)
    colorNotWorking = (0, 255, 0)
    videoManagerInitTimeoutSec = 5
    def __init__(self, videoManager):
        self.videoManager = videoManager
        self.videoManager.doFlipFrame = False

    def run(self):
        self.videoManager.readNewFrame()
        # if not self.videoManager.isInit():
        #     print('is not init')
        #     return
        while not self.videoManager.cap is None:
            self.videoManager.runDetection()
            rectangles = self.videoManager.findDetections([self.classToDetect])
            for rectangle in rectangles:
                self.videoManager.addRectangle(rectangle.pt1.toTuple(), rectangle.pt2.toTuple(), self.colorNotWorking, self.rectThickness)

            self.videoManager.showImage()
            self.videoManager.writeFrame()
            self.videoManager.readNewFrame()

            cmd = self.videoManager.getKeyPress()
            if cmd == 27: # ESC
                break

        self.videoManager.shutdown()