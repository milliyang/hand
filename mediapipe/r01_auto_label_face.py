import cv2
import mediapipe as mp

import os, time

EXIT_KEY = 'q'

file_ext = ['jpg', 'jpeg']
YOLO_IMAGE_SIZE = (416,416)

YOLO_HUMAN_BODY_ID = 0
YOLO_HUMAN_FACE_ID = 1
YOLO_HUMAN_HAND_ID = 2

id_names = {
    YOLO_HUMAN_BODY_ID: "human",
    YOLO_HUMAN_FACE_ID: "face",
    YOLO_HUMAN_HAND_ID: "hand",
}

image_dir = [
    '/home/leo/myhome/hagrid/download/subsample/train'
]

def id_to_names(id):
    if id in id_names.keys():
        return id_names[id]
    else:
        return f"ID:{id}"


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


def auto_label_face_for_yolo(imagefiles = [], show=False):
    mpFaceDetector = mp.solutions.face_detection
    mpDraw = mp.solutions.drawing_utils

    #cap = cv2.VideoCapture(0)
    with mpFaceDetector.FaceDetection(min_detection_confidence=0.7) as faceDetection:

        for _, imagef in enumerate(imagefiles):
            #print(v)
            frame = cv2.imread(imagef)
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

            faces_label = []

            if (results.detections):
                for id, detection in enumerate(results.detections):
                    # Draw the box around the face with built-in function
                    # mpDraw.draw_detection(image, detection)

                    # The box around the face
                    box = detection.location_data.relative_bounding_box

                    # Scale the box with current cap's width and height
                    newBox = int(box.xmin * width), int(box.ymin * height), int(box.width * width), int(box.height * height)

                    if config["show_image"]:
                        # Draw the scaled box with cv2.rectangle
                        cv2.rectangle(image, newBox, (0, 255, 0), 1)
                        cv2.putText(
                            image, f'Score: {detection.score[0]:.2f}',
                            (newBox[0], newBox[1] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                        )

                    #output yolo bbox fmt:  class,cx,xy,w,h
                    face_info = f"{YOLO_HUMAN_FACE_ID} {box.xmin+box.width/2.0} {box.ymin+box.height/2.0} {box.width} {box.height}\n"
                    faces_label.append(face_info)

            #print(imagef)
            #F:\hagrid\download\subsample\subsample\train\call
            labelfile = imagef.replace("train", "train_labels").replace(".jpg", ".txt")
            ff = open(labelfile)
            strings = ff.readlines()
            ff.close()
            if config["show_image"]:
                for each in strings:
                    yolo_fmt = each.strip()
                    yolo_item = yolo_fmt.split()
                #['0', '0.45230302', '0.2694478', '0.05382926', '0.11273142']
                object_id  = yolo_item[0]
                newBox = None
                if False:
                    #xywh:
                    # python.exe  hagrid_to_yolo.py --bbox_format xywh
                    box_xmin   = float(yolo_item[1])
                    box_ymin   = float(yolo_item[2])
                    box_width  = float(yolo_item[3])
                    box_height = float(yolo_item[4])
                    newBox = int(box_xmin * width), int(box_ymin * height), int(box_width * width), int(box_height * height)
                elif False:
                    #xywh:
                    # python.exe  hagrid_to_yolo.py --bbox_format xyxy
                    box_xmin   = float(yolo_item[1])
                    box_ymin   = float(yolo_item[2])
                    box_width  = float(yolo_item[3]) - float(yolo_item[1])
                    box_height = float(yolo_item[4]) - float(yolo_item[2])
                    newBox = int(box_xmin * width), int(box_ymin * height), int(box_width * width), int(box_height * height)
                elif True:
                    #cxcywh:
                    # python.exe  hagrid_to_yolo.py --bbox_format xyxy
                    box_xmin   = float(yolo_item[1]) - float(yolo_item[3]) / 2.0
                    box_ymin   = float(yolo_item[2]) - float(yolo_item[4]) / 2.0
                    box_width  = float(yolo_item[3])
                    box_height = float(yolo_item[4])
                    newBox = int(box_xmin * width), int(box_ymin * height), int(box_width * width), int(box_height * height)

                names = id_to_names(object_id)
                cv2.rectangle(image, newBox, (0, 255, 120), 1)
                cv2.putText(image, names, (newBox[0], newBox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 120), 2)

            if config["auto_label"]:
                new_labelfile = labelfile.replace(".txt", "_auto_hand.txt")
                file = open(new_labelfile, "w")
                for each in strings:
                    file.write(each)
                for each in faces_label:
                    file.write(each)
                file.close()
                print(new_labelfile)

            if config["show_image"]:
                cv2.imshow('Face detection', image)
                time.sleep(1)
                if (cv2.waitKey(10) & 0xFF == ord(EXIT_KEY)):
                    break

if __name__ == '__main__':
    config = {
        "show_image"        : False,
        "dirs"              : image_dir,
        "auto_label"        : True,     #   xxxx.txt -> xxxx.auto_hand.txt
    }

    config = {
        "show_image"        : True,
        "dirs"              : image_dir,
        "auto_label"        : False,
    }
    images = get_all_image_in_dir(config["dirs"])
    auto_label_face_for_yolo(images, config)
