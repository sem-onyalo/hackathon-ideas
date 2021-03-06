import io
import os
import sys
import glob
import json
import time
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
import xml.etree.ElementTree as xt

# TensorFlow model modules
sys.path.append('../../tensorflow/models/research')

from PIL import Image
from core import Constants, Point, Rectangle
from core import ModelManager, VideoManager
from collections import namedtuple
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

DETECTION_LABEL = "Ottawa Senators" # TODO: change dnn model label to 'Ottawa Senators Logo'

class TeamAllegiance:
    _appName = "teamAllegiance"

    isDebug = False
    isTrain = False
    videoManager = None
    classToDetect = 'logo'
    rectangleThickness = 4
    detectionColor = (0, 0, 255)
    labelScale = 0.5
    labelColor = (0, 0, 0)
    labelThickness = 1
    labelBackColor = (255, 255, 255)
    pascalVocSourceDirectory = ""
    trainingDataDirectory = ""
    annotationsDirectory = ""
    imageSetDirectory = ""
    labelMapFilePath = ""
    workingDirectory = ""
    boundingBoxXmlIndex = -1
    trainAndTestSplitPct = 0
    tfrecordTestFilePath = ""
    tfrecordTrainFilePath = ""

    def __init__(self, args):
        self.isDebug = args.debug
        self.isTrain = args.train

        if self.isTrain:
            self.workingDirectory = args.sourcePath
            self.boundingBoxXmlIndex = args.bnbbox_xml_idx
            self.trainAndTestSplitPct = args.train_test_split
            self.pascalVocSourceDirectory = args.pascal_voc_dir
            self.annotationsDirectory = os.path.join(self.pascalVocSourceDirectory, "Annotations")
            self.imageSetDirectory = os.path.join(self.pascalVocSourceDirectory, "JPEGImages")
            self.trainingDataDirectory = os.path.join(self.workingDirectory, "training/data")
            self.labelMapFilePath = os.path.join(self.trainingDataDirectory, "label_map.pbtxt")
            self.tfrecordTestFilePath = os.path.join(self.trainingDataDirectory, f'test.{args.tfrecord_file_ext}')
            self.tfrecordTrainFilePath = os.path.join(self.trainingDataDirectory, f'train.{args.tfrecord_file_ext}')
        else:
            modelObject = {
                'type': Constants.OBJECT_DETECTION,
                'obj': ModelManager.ModelManager.models[Constants.MODEL_INDEX_SPORTS]
            }

            self.videoManager = VideoManager.VideoManager(Constants.WINDOW_NAME, modelObject, args)

    def pascal_voc_to_csv(self, input_dir):
        annot_list = []
        for file in glob.glob(input_dir + '/*.xml'):
            tree = xt.parse(file)
            root = tree.getroot()
            for element in root.findall('object'):
                item = (root.find('filename').text,
                        int(root.find('size')[0].text),
                        int(root.find('size')[1].text),
                        element[0].text,
                        int(float(element[self.boundingBoxXmlIndex][0].text)),
                        int(float(element[self.boundingBoxXmlIndex][1].text)),
                        int(float(element[self.boundingBoxXmlIndex][2].text)),
                        int(float(element[self.boundingBoxXmlIndex][3].text)))
                annot_list.append(item)
            csv_headers = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
            csv_data = pd.DataFrame(annot_list, columns=csv_headers)

        if self.isDebug:
            csv_data.to_csv(f'{self.workingDirectory}\labels.csv', index=None)
        
        return csv_data

    def split_into_test_and_train(self, labels):
        # group all the labels by filename (image)
        labels_grouped = labels.groupby('filename')
        labels_grouped_list = [labels_grouped.get_group(x) for x in labels_grouped.groups]
        image_count = len(labels_grouped_list)

        # get training count by specified percentage
        train_count = round(image_count * self.trainAndTestSplitPct)

        # generate random numbers for train/test indicies
        train_indicies = np.random.choice(image_count, size=train_count, replace=False)
        test_indicies = np.setdiff1d(list(range(image_count)), train_indicies)

        # create train/test labels from random indicies
        train = pd.concat([labels_grouped_list[i] for i in train_indicies])
        test = pd.concat([labels_grouped_list[i] for i in test_indicies])

        if self.isDebug:
            train.to_csv(f'{self.workingDirectory}\labels-train.csv', index=None)
            test.to_csv(f'{self.workingDirectory}\labels-test.csv', index=None)

        return (test, train)

    def create_tf_example(self, label_group, label_map, image_set_path):
        with tf.gfile.GFile(os.path.join(image_set_path, '{}'.format(label_group.filename)), 'rb') as fid:
            encoded_jpg = fid.read()
            
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = Image.open(encoded_jpg_io)
        width, height = image.size

        filename = label_group.filename.encode('utf8')
        image_format = b'jpg'
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        classes = []

        for index, row in label_group.object.iterrows():
            xmins.append(row['xmin'] / width)
            xmaxs.append(row['xmax'] / width)
            ymins.append(row['ymin'] / height)
            ymaxs.append(row['ymax'] / height)
            classes_text.append(row['class'].encode('utf8'))
            classes.append(label_map[row['class']])

        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(filename),
            'image/encoded': dataset_util.bytes_feature(encoded_jpg),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))
        
        return tf_example

    def create_tfrecord_file(self, labels, label_map_path, image_set_dir, output_path):
        tfrecord_writer = tf.python_io.TFRecordWriter(output_path)
        image_set_path = os.path.join(os.getcwd(), image_set_dir)
        
        grouped_labels = labels.groupby('filename')
        label_data = namedtuple('data', ['filename', 'object'])
        label_map = label_map_util.get_label_map_dict(label_map_path)
        
        grouped_label_data = [label_data(filename, grouped_labels.get_group(x)) 
                for filename, x in zip(grouped_labels.groups.keys(), grouped_labels.groups)]
        
        for label_group in grouped_label_data:
            tf_example = self.create_tf_example(label_group, label_map, image_set_path)
            tfrecord_writer.write(tf_example.SerializeToString())

    def createTrainingData(self, inputDirectory, outputDirectory):
        csvLabels = self.pascal_voc_to_csv(inputDirectory)
        testAndTrainLabels = self.split_into_test_and_train(csvLabels)
        self.create_tfrecord_file(testAndTrainLabels[0], self.labelMapFilePath, self.imageSetDirectory, self.tfrecordTestFilePath)
        self.create_tfrecord_file(testAndTrainLabels[1], self.labelMapFilePath, self.imageSetDirectory, self.tfrecordTrainFilePath)

    def getDetectionHandler(self):
        handler = lambda cols, rows, pt1X, pt1Y, pt2X, pt2Y, className, score : self.labelDetections(cols, rows, pt1X, pt1Y, pt2X, pt2Y, className, score)
        return handler

    def labelDetections(self, cols, rows, pt1X, pt1Y, pt2X, pt2Y, className, score):
        self.videoManager.addRectangle((pt1X, pt1Y), (pt2X, pt2Y), self.detectionColor, self.rectangleThickness)
        
        label = DETECTION_LABEL + ": " + str(int(round(score * 100))) + '%'
        self.videoManager.addLabel(label, pt1X, pt1Y)

    def run(self):
        if self.isTrain:
            self.createTrainingData(self.annotationsDirectory, self.workingDirectory)
        else:
            self.videoManager.readNewFrame()
            while not self.videoManager.img is None:
                cmd = self.videoManager.getKeyPress()
                if cmd == 27: # ESC
                    break

                self.videoManager.runObjectDetection()
                self.videoManager.findObjectsDetected([self.classToDetect], self.getDetectionHandler())

                self.videoManager.showImage()
                self.videoManager.writeFrame()
                self.videoManager.readNewFrame()

            self.videoManager.shutdown()
