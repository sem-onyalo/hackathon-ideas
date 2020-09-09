class WorkTracker:
    videoManager = None
    def __init__(self, videoManager):
        self.videoManager = videoManager

    def run(self):
        while True:
            cmd = self.videoManager.getKeyPress()
            if cmd == 27: # ESC
                break
            self.videoManager.readNewFrame()
            self.videoManager.showImage()
        self.videoManager.shutdown()