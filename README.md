# Hackathon Ideas

## Demo

TODO: Put link to demo video here

## Ideas

### Track Work Time

Keep track of the amount of time you are in your work position using object detection.

#### Dependencies

1. [Python](https://www.python.org/downloads/)

2. [OpevCV](https://opencv.org/releases/)

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

### Identify Sports Team Allegiance

Identify a person's sports team allegiances by detecting sports logos in their images.

1. Gather images with the sports logo you want to detect and annotate the logos using ![CVAT](https://github.com/opencv/cvat)

    ![Annotate using CVAT example](identifyteamallegiance/samples/annotation.png)

2. After annotation is complete export the results in Pascal VOC format.

    ![Export as Pascal VOC](identifyteamallegiance/samples/export.png)

3. Extract the contents of [dnn.zip](identifyteamallegiance/cv/dnn.zip) into a folder called [dnn](identifyteamallegiance/cv/dnn). The directory and file structure should look like the following:

    ```
    cv  
    |-- dnn  
    |   |-- data  
    |   |   |-- training  
    |   |   |   |-- data  
    |   |   |   |-- models  
    |   |   |   |   |-- model  
    |   |   |   |   |   |-- train  
    |   |   |   |   |   |-- ssd_mobilenet_v1_coco.config  
    |   |   |   |-- ssd_mobilenet_v1_coco_2017_11_17  
    |   |   |   |   |-- checkpoint  
    |   |   |   |   |-- frozen_inference_graph.pb  
    |   |   |   |   |-- model.ckpt.data-00000-of-00001  
    |   |   |   |   |-- model.ckpt.index  
    |   |   |   |   |-- model.ckpt.meta  
    |   |   |   |   |-- ssd_mobilenet_v1_coco_2017_11_17.pbtxt  
    ```

4. Create a file called [label_map.pbtxt](identifyteamallegiance/cv/dnn/data/training/data/label_map.pbtxt) in the directory [identifyteamallegiance/cv/dnn/data/training/data/](identifyteamallegiance/cv/dnn/data/training/data/) with the following contents:

    ```
    item {  
        id: 1  
        name: 'logo'  
    }
    ```

5. Create the TFRecord training files required to train your model by running the command below. The `--pascal_voc_dir` option is the location of your Pascal VOC annotation extract from CVAT.  
`python main.py 2 identifyteamallegiance/cv/dnn/data/ --pascal_voc_dir identifyteamallegiance/cv/dnn/data/annotations/pascal-voc-1.1 --bnbbox_xml_idx 2`

5. In command prompt `cd` into your local tensorflow repository, e.g. `cd tensorflow/models/research` and run the command to start model training:   
`python object_detection/model_main.py --pipeline_config_path=C:\path\to\sem-onyalo\hackathon-ideas\identifyteamallegiance\cv\dnn\data\training\models\model\ssd_mobilenet_v1_coco.config --model_dir=C:\path\to\sem-onyalo\hackathon-ideas\identifyteamallegiance\cv\dnn\data\training\models\model\train --num_train_steps=50000 --sample_1_of_n_eval_examples=1 --alsologtostderr`

6. Open a second command prompt window, `cd` into your local tensorflow repository, and run the command below to monitor the training process. Once the monitor process is running you can view the training progress in your browser, default: [http://localhost:6006](http://localhost:6006).   
`tensorboard --logdir=C:\code\sem-onyalo\hackathon-ideas\identifyteamallegiance\cv\dnn\data\training\models\model`  

    ![TensorBoard - Monitoring Training Progress](identifyteamallegiance/samples/tensorboard.png)

### Spot Brand Loyalty

`python main.py 3`
