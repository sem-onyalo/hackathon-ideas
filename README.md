# Hackathon Ideas

## Demo

TODO

## Ideas

### Track Work Time

Keep track of the amount of time you are in your work position using object detection.

#### Instructions

1. Run the track work time app on your video: `python main.py 1 trackworktime/samples/in.mp4`.
2. Press `p` to pause the video at the point you want to define the work position.
3. The work position will be a rectangle. You will need to define two points to define that rectangle. On the paused video click where the top left corner of the rectangle will be then click where the bottom right corner will be (see red dots in the image below). 

![Work position selection points](trackworktime/samples/work-position-selection-points.png)

4. Press `ESC` to exit the video.
5. The coordinates of the points you clicked will be printed in the console in the following format: `(pt1, pt2)`. 

![Print mouse click positions](trackworktime/samples/print-click-points.png)

6. Update the `workTracker.workPosition` value in the file [app.settings.json](app.settings.json) based on the points you clicked in the following format: `pt1-top-left,pt2-top-left,pt1-bottom-right,pt2-bottom-right`.
7. Run the track work time app on your video again to see how the work position is matched with the person detected. You may need to tweek the work position value in [app.settings.json](app.settings.json).
8. The total time spent in the work position will be printed in the console once the app exits.

![Print time spent working](trackworktime/samples/print-time-spent-working.png)

### Identify Sports Allegiance

TODO: put description here

`python main.py 2`

1. Annotate using CVAT

![Annotate using CVAT example](identifyteamallegiance/samples/annotation.png)

2. Download Pascal VOC 

![Export as Pascal VOC](identifyteamallegiance/samples/export.png)

3. Create the file [label_map.pbtxt](identifyteamallegiance/cv/dnn/data/training/data/label_map.pbtxt) with the following contents:

    item {
        id: 1
        name: 'logo'
    }

### Spot Brand Loyalty

`python main.py 3`
