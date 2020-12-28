import set_model
import tensorflow as tf
import cv2
import numpy as np


class model:
    def __init__(self):
        detection_graph, self.category_index = set_model.set_model(
            "ssd_mobilenet_v1_coco_2018_01_28", "mscoco_label_map.pbtxt"
        )
        self.sess = tf.InteractiveSession(graph=detection_graph)
        self.image_tensor = detection_graph.get_tensor_by_name("image_tensor:0")
        self.detection_boxes = detection_graph.get_tensor_by_name("detection_boxes:0")
        self.detection_scores = detection_graph.get_tensor_by_name("detection_scores:0")
        self.detection_classes = detection_graph.get_tensor_by_name(
            "detection_classes:0"
        )
        self.num_detections = detection_graph.get_tensor_by_name("num_detections:0")

    def get_category_index(self):
        return self.category_index

    def detect_pedestrians(self, frame):
        input = frame

        image_expanded = np.expand_dims(input, axis=0)
        (box, score, classes, number) = self.sess.run(
            [
                self.detection_boxes,
                self.detection_scores,
                self.detection_classes,
                self.num_detections,
            ],
            feed_dict={self.image_tensor: image_expanded},
        )

        classes = np.squeeze(classes).astype(np.int32)
        box = np.squeeze(box)
        score = np.squeeze(score)
        pedestrian_threshold = 0.35
        all_boxes = []
        all_pedestrians = 0
        for i in range(int(number[0])):
            if classes[i] in self.category_index.keys():
                class_name = self.category_index[classes[i]]["name"]
                # print(class_name)
                if class_name == "person" and score[i] > pedestrian_threshold:
                    all_pedestrians += 1
                    score_pedestrian = score[i]
                    all_boxes.append(box[i])

        return all_boxes, all_pedestrians
