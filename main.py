import argparse
from core import ModelManager, VideoManager
from trackworktime import WorkTracker
from identifyteamallegiance import TeamAllegiance

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("optionId", type=int, help="The option to run: \
        1 - Track Work Time \
        2 - Spot Brand Loyalty \
        3 - Identify Sports Allegiance")
    parser.add_argument("sourcePath", type=str, help="The path to the video being processed.")
    parser.add_argument("--out", type=str, help="The path to the output the processed video")
    parser.add_argument("--res", type=str, default="1280,720", help="The resolution of the video. Default is \"1280,720\"")
    parser.add_argument("--reps", type=int, default=5, help="The number of player reps per spot challenge game. Default is 5.")
    parser.add_argument("--train", action="store_true", help="Run model training")
    parser.add_argument("--score_threshold", type=float, default=0.3, help="Only detections with a probability of correctness above the specified threshold")
    parser.add_argument("--pascal_voc_dir", type=str, help="The directory of the Pascal VOC extract files")
    parser.add_argument("--bnbbox_xml_idx", type=int, default=4, help="The array index of bounding box info in the Pascal VOC xml files.")
    parser.add_argument("--train_test_split", type=float, default=0.6, help="The percentage of test/train split")
    parser.add_argument("--tfrecord_file_ext", type=str, default="record-00000-of-00010")
    args = parser.parse_args()
    
    res = list(map(int, args.res.split(',')))

    if args.optionId == 1:
        model = ModelManager.ModelManager.models[0]
        videoManager = VideoManager.VideoManager('Hackathon Ideas', args.sourcePath, args.out, res[0], res[1], model, args.score_threshold)
        app = WorkTracker.WorkTracker(videoManager)
    elif args.optionId == 2:
        app = TeamAllegiance.TeamAllegiance(args)
    else:
        raise RuntimeError('Invalid option:', args.optionId)

    app.run()
    