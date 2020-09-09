import argparse
from core import ModelManager, VideoManager
from trackworktime import WorkTracker

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("optionId", type=int, help="The option to run: \
        1 - Track Work Time \
        2 - Spot Brand Loyalty \
        3 - Identify Sports Allegiance")
    parser.add_argument("--res", type=str, default="1280,720", help="The resolution of the video. Default is \"1280,720\"")
    parser.add_argument("--video_source", type=int, default=0, help="The index of the video source. Default is 0.")
    parser.add_argument("--reps", type=int, default=5, help="The number of player reps per spot challenge game. Default is 5.")
    parser.add_argument("--score_threshold", type=float, default=0.3, help="Only detections with a probability of correctness above the specified threshold")
    args = parser.parse_args()
    
    res = list(map(int, args.res.split(',')))

    if args.optionId == 1:
        model = ModelManager.ModelManager.models[0]
        videoManager = VideoManager.VideoManager(args.video_source, 'Track Work Time', res[0], res[1], model, args.score_threshold)
        app = WorkTracker.WorkTracker(videoManager)
    else:
        raise RuntimeError('Invalid option:', args.optionId)

    app.run()
