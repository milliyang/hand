import cv2
import mediapipe as mp
import os, time


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


EXIT_KEY = 'q'

file_ext = ['jpg', 'jpeg']
YOLO_IMAGE_SIZE = (320,320)

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

linefont = cv2.FONT_HERSHEY_SIMPLEX

def id_to_names(id):
    id = int(id)
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

def auto_label_face_for_yolo(imagefiles = [], config = {}):
    mpFaceDetector = mp.solutions.face_detection
    mpDraw = mp.solutions.drawing_utils
    mp_face_cfg = {
        "min_detection_confidence" : config['face_detect_thresh'], #0.5
        "model_selection" : 1,      # 1,near,far; 0,near;
    }
    mp_pose_cfg = {
        "static_image_mode"         : True,
        "model_complexity"          : 2,    #0,1,2
        "enable_segmentation"       : False,
        "min_detection_confidence"  : config['pose_detect_thresh'], #0.5
        "upper_body_only"           : False,
        "enable_segmentation"       : False,
        "smooth_segmentation"       : False,
        "min_tracking_confidence"   : 0.5,
    }
    pose_detection = mp_pose.Pose(mp_pose_cfg)

    #cap = cv2.VideoCapture(0)
    with mpFaceDetector.FaceDetection(**mp_face_cfg) as faceDetection:

        for _, imagef in enumerate(imagefiles):
            frame = cv2.imread(imagef)
            frame = cv2.resize(frame, YOLO_IMAGE_SIZE)
            height, width, _  = frame.shape

            # Flip the frame horizontally
            # frame = cv2.flip(frame, 1)

            # Convert the color for the program to process
            # Since cv2 uses BGR and mediapipe uses RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = faceDetection.process(image)

            # Convert it back for displaying after processing
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            faces_label = []

            if config["pose_detect"]:
                pose_ret = pose_detection.process(image)
                #draw pose
                mp_drawing.draw_landmarks(
                image,
                pose_ret.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                if pose_ret.pose_landmarks:
                    #https://blog.csdn.net/weixin_43229348/article/details/120541448
                    #https://developers.google.cn/android/reference/com/google/mlkit/vision/pose/PoseLandmark
                    eye = (pose_ret.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE].x * width,
                           pose_ret.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE].y * height)
                    shoulder = (pose_ret.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * width,
                               pose_ret.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * height)
                    hip = (pose_ret.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * width,
                           pose_ret.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * height)
                    knee = (pose_ret.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x * width,
                            pose_ret.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y * height)
                    foot = (pose_ret.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x * width,
                            pose_ret.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y * height)
                    pose_cc = (250, 0, 0)
                    cv2.putText( image, f'hip',  (int(hip[0]), int(hip[1])), linefont, 0.5, pose_cc, 1)
                    cv2.putText( image, f'shoulder',  (int(shoulder[0]), int(shoulder[1])), linefont, 0.5, pose_cc, 1)
                    cv2.putText( image, f'eye',  (int(eye[0]), int(eye[1])), linefont, 0.5, pose_cc, 1)
                    cv2.putText( image, f'knee', (int(knee[0]), int(knee[1])), linefont, 0.5, pose_cc, 1)
                    cv2.putText( image, f'foot', (int(foot[0]), int(foot[1])), linefont, 0.5, pose_cc, 1)
                    #print(f"foot:{foot} frame.shape:{frame.shape}")

            if config['face_detect']:
                if (results.detections):
                    for _, detection in enumerate(results.detections):
                        # mpDraw.draw_detection(image, detection)   #built-in function
                        # The box around the face
                        box = detection.location_data.relative_bounding_box

                        if config["show_image"]:
                            face_box = int(box.xmin * width), int(box.ymin * height), int(box.width * width), int(box.height * height)
                            face_cc = (0, 0, 250)
                            cv2.rectangle(image, face_box, face_cc, 1)
                            cv2.putText( image, f'face:{detection.score[0]:.2f}', (face_box[0]+40, face_box[1] - 20), linefont, 0.5, face_cc, 1 )

                        #output yolo bbox fmt:  class,cx,xy,w,h
                        face_info = f"{YOLO_HUMAN_FACE_ID} {box.xmin+box.width/2.0} {box.ymin+box.height/2.0} {box.width} {box.height}\n"
                        faces_label.append(face_info)

            #print(imagef)
            #image: /home/leo/myhome/hagrid/download/subsample/train
            #label: /home/leo/myhome/hagrid/download/subsample/train_labels
            labelfile = imagef.replace("train", "train_labels").replace(".jpg", ".txt")
            ff = open(labelfile)
            strings = ff.readlines()
            ff.close()
            if config["show_image"]:
                for yolo_fmt in strings:
                    items = yolo_fmt.strip().split()
                    object_id  = items[0]
                    #['0', '0.45230302', '0.2694478', '0.05382926', '0.11273142']

                    #cxcywh:
                    # python.exe  hagrid_to_yolo.py --bbox_format cxcywh
                    box_xmin   = float(items[1]) - float(items[3]) / 2.0
                    box_ymin   = float(items[2]) - float(items[4]) / 2.0
                    box_width  = float(items[3])
                    box_height = float(items[4])
                    abox = int(box_xmin * width), int(box_ymin * height), int(box_width * width), int(box_height * height)

                    names = id_to_names(object_id)
                    cv2.rectangle(image, abox, (0, 255, 120), 1)
                    cv2.putText(image, names, (abox[0], abox[1]-8), linefont, 0.5, (0, 255, 120), 1)

            if config["auto_label"]:
                new_labelfile = labelfile.replace(".txt", "_mp_hand.txt")
                file = open(new_labelfile, "w")
                for each in strings:
                    file.write(each)
                for each in faces_label:
                    file.write(each)
                file.close()
                print(new_labelfile)

            if config["show_image"]:
                cv2.imshow('Face detection', image)
                wait_time = config["show_image_wait"]
                if wait_time > 0:
                    time.sleep(wait_time)
                if (cv2.waitKey(10) & 0xFF == ord(EXIT_KEY)):
                    break

if __name__ == '__main__':
    config = {
        "show_image"            : False,
        "show_image_wait"       : 0,
        "dirs"                  : image_dir,
        "face_detect"           : True,
        "face_detect_thresh"    : 0.4,
        "pose_detect"           : True,
        "pose_detect_thresh"    : 0.4,
        "auto_label"            : True,     #   xxxx.jpg -> xxxx.auto_hand.txt
    }

    DEBUG = True
    if DEBUG:
        config = {
            "show_image"            : True,
            "show_image_wait"       : 1.2,
            "dirs"                  : image_dir,
            "face_detect"           : True,
            "face_detect_thresh"    : 0.2,
            "pose_detect"           : True,
            "pose_detect_thresh"    : 0.2,
            "auto_label"            : False,
        }
    images = get_all_image_in_dir(config["dirs"])
    auto_label_face_for_yolo(images, config)
