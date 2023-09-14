#!wget -q -O efficientdet.tflite -q https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite

import cv2
import numpy as np

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red

def visualize(
    image,
    detection_result
) -> np.ndarray:
  """Draws bounding boxes on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """
  for detection in detection_result.detections:
    category = detection.categories[0]
    category_name = category.category_name
    if category_name != 'person':
      continue

    # Draw bounding_box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

    # Draw label and score
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (MARGIN + bbox.origin_x,
                     MARGIN + ROW_SIZE + bbox.origin_y)
    cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)
  return image

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an ObjectDetector object.
base_options = python.BaseOptions(model_asset_path='efficientdet_lite2.tflite')
options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.40)
detector = vision.ObjectDetector.create_from_options(options)

# STEP 3: Load the input image.
#image = mp.Image.create_from_file('/mnt/214cd1a6-9d1b-4b2a-9770-425b64e6884f/myhome/WIDER_train/train/4--Dancing/4_Dancing_Dancing_4_108.jpg')
#image = mp.Image.create_from_file('/mnt/214cd1a6-9d1b-4b2a-9770-425b64e6884f/myhome/WIDER_train/train/44--Aerobics/44_Aerobics_Aerobics_44_8.jpg')
#image = mp.Image.create_from_file('/mnt/214cd1a6-9d1b-4b2a-9770-425b64e6884f/myhome/WIDER_train/train/44--Aerobics/44_Aerobics_Aerobics_44_2.jpg')
image = mp.Image.create_from_file("/home/leo/myhome/hand/mediapipe/object_detection.py")

# STEP 4: Detect objects in the input image.
detection_result = detector.detect(image)

# STEP 5: Process the detection result. In this case, visualize it.
image_copy = np.copy(image.numpy_view())
annotated_image = visualize(image_copy, detection_result)
rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
cv2.imshow('img', rgb_annotated_image)
cv2.waitKey(1000*10)

# %%



