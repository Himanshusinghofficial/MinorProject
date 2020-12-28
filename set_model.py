
import glob, os, tarfile, urllib
import tensorflow as tf
from utils import label_map_util


def set_model(model_name, label_name):
    model_count = 0

    for file in glob.glob("*"):
        if file == model_name:
            model_count = 1

    # What model to download.
    model_name = model_name
    model_file = model_name + ".tar.gz"
    download_base = "http://download.tensorflow.org/models/object_detection/"

    path_to_check = model_name + "/frozen_inference_graph.pb"

    path_to_label = os.path.join("data", label_name)

    number_classes = 90

    # Download Model if it has not been downloaded yet
    if model_count == 0:
        openers = urllib.request.URLopener()
        openers.retrieve(download_base + model_file, model_file)
        target_file = tarfile.open(model_file)
        for file in target_file.getmembers():
            file_name = os.path.basename(file.name)
            if "frozen_inference_graph.pb" in file_name:
                target_file.extract(file, os.getcwd())


    # Load a (frozen) Tensorflow model into memory.
    detection_of_graph = tf.Graph()
    with detection_of_graph.as_default():
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_check, "rb") as fid:
            serialize_graph = fid.read()
            graph_def.ParseFromString(serialize_graph)
            tf.import_graph_def(graph_def, name="")

    # Loading label map
    labels_map = label_map_util.load_labelmap(path_to_label)
    category = label_map_util.convert_label_map_to_categories(
        labels_map, max_num_classes=number_classes, use_display_name=True
    )
    category_of_index = label_map_util.create_category_index(category)

    return detection_of_graph, category_of_index
