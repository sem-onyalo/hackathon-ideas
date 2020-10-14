import argparse
from core import ModelManager, VideoManager
from trackworktime import WorkTracker
from identifyteamallegiance import TeamAllegiance
from spotbrandloyalty import BrandOcr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("optionId", type=int, help="The option to run: \
        1 - Track Work Time \
        2 - Spot Brand Loyalty \
        3 - Identify Sports Team Allegiance")
    parser.add_argument("sourcePath", type=str, help="The path to the video being processed.")
    parser.add_argument("--out", type=str, help="The path to the output the processed video")
    parser.add_argument("--res", type=str, default="1280,720", help="The resolution of the video. Default is \"1280,720\"")
    parser.add_argument("--train", action="store_true", help="Run model training")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--hide", action="store_true", help="Hide the video feed")
    parser.add_argument("--score_threshold", type=float, default=0.3, help="Only detections with a probability of correctness above the specified threshold")
    parser.add_argument("--pascal_voc_dir", type=str, help="The directory of the Pascal VOC extract files")
    parser.add_argument("--bnbbox_xml_idx", type=int, default=4, help="The array index of bounding box info in the Pascal VOC xml files.")
    parser.add_argument("--train_test_split", type=float, default=0.6, help="The percentage of test/train split")
    parser.add_argument("--tfrecord_file_ext", type=str, default="record-00000-of-00010")
    parser.add_argument("--ocr_preproc_blur", action="store_true", help="OCR preprocessing - median blur")
    parser.add_argument("--ocr_preproc_thresh", action="store_true", help="OCR preprocessing - threshold")
    parser.add_argument("--ocr_preproc_res", type=str, default="320,320", help="The preprocessing image resolution.")
    parser.add_argument("--ocr_lang", type=str, default="eng", help="OCR language")
    parser.add_argument("--ocr_oem", type=int, default=1, help="OCR engine mode")
    parser.add_argument("--ocr_psm", type=int, default=7, help="OCR page segmentation mode")
    parser.add_argument("--ocr_padding", type=float, default=0.0, help="Amount of padding as a percent to add to text recognition bounding box")
    parser.add_argument("--fzm_threshold", type=float, default=0.8, help="Only show fuzzy string matches equal to or above this threshold")
    args = parser.parse_args()

    if args.optionId == 1:
        app = WorkTracker.WorkTracker(args)
    elif args.optionId == 2:
        app = BrandOcr.BrandOcr(args)
    elif args.optionId == 3:
        app = TeamAllegiance.TeamAllegiance(args)
    else:
        raise RuntimeError('Invalid option:', args.optionId)

    app.run()
    