import cv2
import mediapipe as mp

import os, time

EXIT_KEY = 'q'

file_ext = ['jpg', 'jpeg']
YOLO_IMAGE_SIZE = (416,416)

YOLO_HUMAN_FACE_ID = 1

image_dir = [
    'F:\\hagrid\\download\\subsample\\subsample\\train'
]

def get_images_in_current_dir(dir):
    if not os.path.isdir(dir):
        return None
    files = os.listdir(dir)
    select_files = []
    for onefile in files:
        ext = onefile.split(".")[-1]
        if ext in file_ext:
            select_files.append(os.path.join(dir, onefile))
    return select_files

def get_all_image_in_dir(dir_list =[]):
    images = []
    for one_dir in dir_list:
        for entry in os.listdir(one_dir):
            path = os.path.join(one_dir, entry)
            if os.path.isdir(path):
                imgs = get_images_in_current_dir(path)
                images.extend(imgs)
                print(len(imgs), "totals:", len(images), path)
        break #quick debug
    return images

def auto_label_face_for_yolo(imagefiles = []):
    mpFaceDetector = mp.solutions.face_detection
    mpDraw = mp.solutions.drawing_utils

    #cap = cv2.VideoCapture(0)
    with mpFaceDetector.FaceDetection(min_detection_confidence=0.7) as faceDetection:

        for _, v in enumerate(imagefiles):
            #print(v)
            frame = cv2.imread(v)
            frame = cv2.resize(frame, YOLO_IMAGE_SIZE)

            size = frame.shape
            # Camera's width & height
            width  = size[1]
            height = size[0]

            # Flip the frame horizontally
            #frame = cv2.flip(frame, 1)

            # Convert the color for the program to process
            # Since cv2 uses BGR and mediapipe uses RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = faceDetection.process(image)

            # Convert it back for displaying after processing
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if (results.detections):
                for id, detection in enumerate(results.detections):
                    # Draw the box around the face with built-in function
                    # mpDraw.draw_detection(image, detection)

                    # The box around the face
                    box = detection.location_data.relative_bounding_box
                    
                    # Scale the box with current cap's width and height
                    newBox = int(box.xmin * width), int(box.ymin * height), int(box.width * width), int(box.height * height)

                    # Draw the scaled box with cv2.rectangle
                    cv2.rectangle(image, newBox, (0, 255, 0), 1)

                    cv2.putText(
                        image, f'Score: {detection.score[0]:.2f}',
                        (newBox[0], newBox[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                    )

                    #output file, yolo fmt class, bbox info


            cv2.imshow('Face detection', image)
            time.sleep(1)

            if (cv2.waitKey(10) & 0xFF == ord(EXIT_KEY)):
                break


if __name__ == '__main__':
    
    images = get_all_image_in_dir(image_dir)
    auto_label_face_for_yolo(images)