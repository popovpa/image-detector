import os
import sys

# Import everything needed to edit video clips
# import moviepy.editor as mp
import cv2
import numpy as np
import tensorflow as tf

# if tf.__version__ < '1.4.0':
#     raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

# This is needed to display the images.

# This is needed since the notebook is stored in the object_detection folder.

DATA_PATH = '/_data/tf/object_detect/'

sys.path.append(DATA_PATH)

from utils import label_map_util
from utils import visualization_utils as vis_util
# from utils import ops as utils_ops

MODEL_NAME = 'mask_rcnn_resnet50_atrous_coco_2018_01_28'
# MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'
MODEL_FILE = DATA_PATH + MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = DATA_PATH + MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# opener = urllib.request.URLopener()
# opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
# tar_file = tarfile.open(MODEL_FILE)
# for file in tar_file.getmembers():
#    file_name = os.path.basename(file.name)
#    if 'frozen_inference_graph.pb' in file_name:
#        tar_file.extract(file, os.getcwd())

with tf.device("/device:CPU:0"):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
    # (im_width, im_height) = image.size
    return np.array(image).reshape(
        (image.shape)).astype(np.uint8)


# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3)]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

with detection_graph.as_default():
    with tf.device("/device:CPU:0"):
        with tf.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # masks tensor
            detection_masks_tensor = detection_graph.get_tensor_by_name('detection_masks:0')

            # clip = mp.VideoFileClip("test_images/video.mp4").subclip(2, 5).resize(height=180)

            cap = cv2.VideoCapture(0)
            cap.set(3, 640 * 1)
            cap.set(4, 480 * 1)
            cnt = 0

            while True:
                # Capture frame-by-frame
                ret, frame = cap.read()

                # Our operations on the frame come here

                # img_cv = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                image_np = load_image_into_numpy_array(frame)

                image_np_expanded = np.expand_dims(image_np, axis=0)

                # detection_boxes = tf.squeeze(detection_boxes, [0])
                #detection_masks = tf.squeeze(detection_masks_tensor, [0])
                # # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size
                # real_num_detection = tf.cast(num_detections[0], tf.int32)
                # detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                # detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                # # detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                # #     detection_masks, detection_boxes, image_np.shape[0], image_np.shape[1])
                # detection_masks_reframed = tf.cast(
                #     tf.greater(1.6, 0.5), tf.uint8)
                # # Follow the convention by adding back the batch dimension
                # detection_masks = tf.expand_dims(detection_masks_reframed, 0)

                # if cnt % 5 == 0:
                #(boxes, scores, classes, num, masks) = sess.run(
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],  # , detection_masks],
                    feed_dict={image_tensor: image_np_expanded})

                #detection_masks = tf.squeeze(detection_masks, [0])

                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    frame,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    # masks overlay
                    #instance_masks=masks[0],
                    use_normalized_coordinates=True,
                    line_thickness=2)

                cnt += 1

                # Display the resulting frame
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # When everything done, release the capture
            cap.release()
            cv2.destroyAllWindows()
