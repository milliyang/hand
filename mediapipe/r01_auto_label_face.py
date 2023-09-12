import cv2
import mediapipe as mp
import os, time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

EXIT_KEY = 'q'

file_ext = ['jpg', 'jpeg']
YOLO_IMAGE_SIZE = (410,410)

YOLO_HUMAN_HUMAN_ID   = 0
YOLO_HUMAN_FACE_ID    = 1
YOLO_HUMAN_HAND_ID    = 2
YOLO_HUMAN_BODY_ID    = 3
YOLO_HUMAN_FOOT_ID    = 4
#
YOLO_HUMAN_HAND_ONE   = 5
YOLO_HUMAN_HAND_TWO   = 6     #peace, peace_inv, two_up
YOLO_HUMAN_HAND_THREE = 7
YOLO_HUMAN_HAND_FOUR  = 8
YOLO_HUMAN_HAND_FIVE  = 9     #five,stop        -->Larger
YOLO_HUMAN_HAND_OK    = 10    #                 -->Smaller
YOLO_HUMAN_HAND_LIKE  = 11
YOLO_HUMAN_HAND_CALL  = 12
YOLO_HUMAN_HAND_FIST  = 13
#
YOLO_DOG_ID      = 14
YOLO_CAT_ID      = 15
YOLO_BIRD_ID     = 16
YOLO_HORSE_ID    = 17
YOLO_SHEEP_ID    = 18
YOLO_CAR_ID      = 19
YOLO_COW_ID      = 20

id_names = {
    YOLO_HUMAN_HUMAN_ID   : "human",
    YOLO_HUMAN_FACE_ID    : "face",
    YOLO_HUMAN_HAND_ID    : "hand",
    YOLO_HUMAN_BODY_ID    : "body",
    YOLO_HUMAN_FOOT_ID    : "foot",
    #
    YOLO_HUMAN_HAND_ONE   : "one",
    YOLO_HUMAN_HAND_TWO   : "two",
    YOLO_HUMAN_HAND_THREE : "three",
    YOLO_HUMAN_HAND_FOUR  : "four",
    YOLO_HUMAN_HAND_FIVE  : "five",
    YOLO_HUMAN_HAND_OK    : "ok",
    YOLO_HUMAN_HAND_LIKE  : "like",
    YOLO_HUMAN_HAND_CALL  : "call",
    YOLO_HUMAN_HAND_FIST  : "fist",
    #
    YOLO_DOG_ID      : "dog",
    YOLO_CAT_ID      : "cat",
    YOLO_BIRD_ID     : "bird",
    YOLO_HORSE_ID    : "horse",
    YOLO_SHEEP_ID    : "sheep",
    YOLO_CAR_ID      : "car",
    YOLO_COW_ID      : "cow",
}

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

def get_all_image_in_dir(dir_list =[], subsample=100000):
    images = []
    for one_dir in dir_list:
        for entry in os.listdir(one_dir):
            path = os.path.join(one_dir, entry)
            if os.path.isdir(path):
                imgs = get_images_in_current_dir(path)

                if (len(imgs) >= subsample):
                    imgs = imgs[0:subsample]

                images.extend(imgs)
                print(len(imgs), "totals:", len(images), path)
        break #quick debug
    return images

def get_rect_from_landmarks(id, landmarks, alist:list, margin=0):
    all_x = []
    all_y = []

    for mark in alist:
        x = landmarks[mark].x
        y = landmarks[mark].y
        if (x >= 0) and (x <= 1.0) and (y >= 0) and (y <= 1.0):
            all_x.append(x)
            x0 = max(x - margin, 0.0)
            x1 = min(x + margin, 1.0)
            all_x.append(x0)
            all_x.append(x1)

            all_y.append(y)
            y0 = max(y - margin, 0.0)
            y1 = min(y + margin, 1.0)
            all_y.append(y0)
            all_y.append(y1)

    if len(all_x) > 0 and len(all_y) > 0:
        core = (id, id_names[id], (min(all_x), min(all_y), max(all_x), max(all_y)))
        return core
    else:
        return None

def body_info_from_landmark(landmarks):
    INDEX = mp_pose.PoseLandmark
    margin_x = landmarks[INDEX.LEFT_EYE].x - landmarks[INDEX.RIGHT_EYE].x
    margin_y = landmarks[INDEX.LEFT_EYE].y - landmarks[INDEX.RIGHT_EYE].y
    head_margin = max(margin_x, margin_y)
    #print(margin)

    foot_margin_x = landmarks[INDEX.LEFT_HEEL].x - landmarks[INDEX.LEFT_FOOT_INDEX].x
    foot_margin_y = landmarks[INDEX.LEFT_HEEL].y - landmarks[INDEX.LEFT_FOOT_INDEX].y
    foot_margin = max(foot_margin_x, foot_margin_y) / 2.0

    head_face_list   = [INDEX.NOSE, INDEX.LEFT_EYE, INDEX.RIGHT_EYE, INDEX.LEFT_EAR, INDEX.RIGHT_EAR, INDEX.MOUTH_LEFT, INDEX.MOUTH_RIGHT]
    body_marks_list  = [INDEX.LEFT_SHOULDER, INDEX.RIGHT_SHOULDER, INDEX.LEFT_HIP, INDEX.RIGHT_HIP, INDEX.LEFT_KNEE, INDEX.RIGHT_KNEE]
    #
    hand_left_list   = [INDEX.LEFT_WRIST, INDEX.LEFT_PINKY, INDEX.LEFT_INDEX, INDEX.LEFT_THUMB]
    hand_right_list  = [INDEX.RIGHT_WRIST, INDEX.RIGHT_PINKY, INDEX.RIGHT_INDEX, INDEX.RIGHT_THUMB]
    foot_left_list   = [INDEX.LEFT_ANKLE, INDEX.LEFT_HEEL, INDEX.LEFT_FOOT_INDEX]
    foot_right_list  = [INDEX.RIGHT_ANKLE, INDEX.RIGHT_HEEL, INDEX.RIGHT_FOOT_INDEX]

    head        = get_rect_from_landmarks(YOLO_HUMAN_FACE_ID, landmarks, head_face_list, head_margin)
    body        = get_rect_from_landmarks(YOLO_HUMAN_BODY_ID, landmarks, body_marks_list, 0)
    hand_left   = get_rect_from_landmarks(YOLO_HUMAN_HAND_ID, landmarks, hand_left_list,  0)
    hand_right  = get_rect_from_landmarks(YOLO_HUMAN_HAND_ID, landmarks, hand_right_list, 0)
    foot_left   = get_rect_from_landmarks(YOLO_HUMAN_FOOT_ID, landmarks, foot_left_list,  foot_margin)
    foot_right  = get_rect_from_landmarks(YOLO_HUMAN_FOOT_ID, landmarks, foot_right_list, foot_margin)

    rects = []
    #if head: rects.append(head)        #ignore head

    if body: rects.append(body)
    if hand_left: rects.append(hand_left)
    if hand_right: rects.append(hand_right)
    if foot_left: rects.append(foot_left)
    if foot_right: rects.append(foot_right)

    print(rects)

    return rects

def get_object_detector(thresh=0.40):
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    #
    base_options = python.BaseOptions(model_asset_path='efficientdet_lite2.tflite')
    options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=thresh)
    detector = vision.ObjectDetector.create_from_options(options)
    return detector

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

    object_detection = get_object_detector(config['person_detect_thresh'])

    #cap = cv2.VideoCapture(0)
    with mpFaceDetector.FaceDetection(**mp_face_cfg) as faceDetection:

        for _, imagef in enumerate(imagefiles):
            frame = cv2.imread(imagef)
            frame = cv2.resize(frame, YOLO_IMAGE_SIZE)
            height, width, _  = frame.shape

            # Flip the frame horizontally
            # frame = cv2.flip(frame, 1)

            # cv2 uses BGR and mediapipe uses RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_results = faceDetection.process(image_rgb)

            # Convert it back for displaying after processing
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            faces_label = []
            bodys_label = []
            person_label = []

            if config["person_detect"]:
                #print(dir(mp.Image))
                #imageL = mp.Image.create_from_file(imagef)
                #detection_result = object_detection.detect(imageL)
                rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                detection_result = object_detection.detect(rgb_frame)
                for detection in detection_result.detections:
                    category = detection.categories[0]
                    category_name = category.category_name
                    if category_name != 'person':
                        continue

                    pson_cc = (250, 100, 0)

                    # Draw bounding_box
                    bbox = detection.bounding_box
                    pt_start = bbox.origin_x, bbox.origin_y
                    pt_end = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
                    cv2.rectangle(image, pt_start, pt_end, pson_cc, 2)

                    # Draw label and score
                    probability = round(category.score, 2)
                    text_info = category_name + ' (' + str(probability) + ')'
                    pt_text_location = (bbox.origin_x, max(bbox.origin_y-10, 0))
                    cv2.putText(image, text_info, pt_text_location, linefont, 0.5, pson_cc, 1)

                    #output yolo bbox fmt:  class,cx,xy,w,h
                    pson_cxcywh = (bbox.origin_x + bbox.width/2.0, bbox.origin_y + bbox.height/2.0, bbox.width, bbox.height)
                    pson_info = f"{YOLO_HUMAN_HUMAN_ID} {pson_cxcywh[0]} {pson_cxcywh[1]} {pson_cxcywh[2]} {pson_cxcywh[3]} \n"
                    person_label.append(pson_info)

            if config["pose_detect"]:
                pose_result = pose_detection.process(image_rgb)
                #draw pose
                mp_drawing.draw_landmarks(
                image,
                pose_result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                if pose_result.pose_landmarks :
                    #https://blog.csdn.net/weixin_43229348/article/details/120541448
                    #https://developers.google.cn/android/reference/com/google/mlkit/vision/pose/PoseLandmark
                    #eye = (pose_result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE].x * width,
                    #       pose_result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE].y * height)
                    #shoulder = (pose_result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * width,
                    #           pose_result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * height)
                    #hip = (pose_result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * width,
                    #       pose_result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * height)
                    #knee = (pose_result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x * width,
                    #        pose_result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y * height)
                    #foot = (pose_result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x * width,
                    #        pose_result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y * height)
                    #pose_cc = (250, 0, 0)
                    #cv2.putText( image, f'hip',  (int(hip[0]), int(hip[1])), linefont, 0.5, pose_cc, 1)
                    #cv2.putText( image, f'shoulder',  (int(shoulder[0]), int(shoulder[1])), linefont, 0.5, pose_cc, 1)
                    #cv2.putText( image, f'eye',  (int(eye[0]), int(eye[1])), linefont, 0.5, pose_cc, 1)
                    #cv2.putText( image, f'knee', (int(knee[0]), int(knee[1])), linefont, 0.5, pose_cc, 1)
                    #cv2.putText( image, f'foot', (int(foot[0]), int(foot[1])), linefont, 0.5, pose_cc, 1)
                    ##print(f"foot:{foot} frame.shape:{frame.shape}")

                    body_info = body_info_from_landmark(pose_result.pose_landmarks.landmark)
                    for a_rect in body_info:
                        yolo_id_idx, yolo_name, a_rect = a_rect
                        #print("body_rect:", body_rect)
                        #
                        a_xywh = (a_rect[0]*width, a_rect[1]*height, (a_rect[2]-a_rect[0])*width, (a_rect[3]-a_rect[1])*height)
                        a_xywh = int(a_xywh[0]), int(a_xywh[1]), int(a_xywh[2]), int(a_xywh[3])
                        body_cc = (250, 250, 0)
                        cv2.rectangle(image, a_xywh, body_cc, 1)
                        cv2.putText( image, yolo_name, (a_xywh[0]+10, a_xywh[1]-10), linefont, 0.5, body_cc, 2)

                        #output yolo bbox fmt:  class,cx,xy,w,h
                        yolo_cxcywh = ((a_rect[0]+a_rect[2])/2.0, (a_rect[1]+a_rect[3])/2.0, a_rect[2]-a_rect[0], a_rect[3]-a_rect[1])
                        yolo_info = f"{yolo_id_idx} {yolo_cxcywh[0]} {yolo_cxcywh[1]} {yolo_cxcywh[2]} {yolo_cxcywh[3]} \n"
                        bodys_label.append(yolo_info)

            if config['face_detect']:
                if (face_results.detections):
                    for _, detection in enumerate(face_results.detections):
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
            if config["show_image"] and config['draw_labels_in_txt_file']:
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
                for each in bodys_label:
                    file.write(each)
                for each in person_label:
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

    image_dir = [
        #'/home/leo/myhome/hagrid/download/subsample/train',
        '/home/leo/hand_fullset/train'
    ]

    config = {
        "show_image"                : False,
        "show_image_wait"           : 0,
        "dirs"                      : image_dir,
        "dirs_subsample_max"        : 1000,
        "face_detect"               : True,
        "face_detect_thresh"        : 0.2,
        "pose_detect"               : True,
        "pose_detect_thresh"        : 0.1,
        "person_detect"             : True,
        "person_detect_thresh"      : 0.30,
        "auto_label"                : True,     #   xxxx.jpg -> xxxx.auto_hand.txt
        "draw_labels_in_txt_file"   : True,     #   draw label(id,x,y,w,h) from txtfile
    }

    DEBUG = 1
    if DEBUG == 0:
        #
        config = {
            "show_image"                : True,
            "show_image_wait"           : 0.6,
            "dirs"                      : image_dir,
            "dirs_subsample_max"        : 1000,
            "face_detect"               : True,
            "face_detect_thresh"        : 0.2,
            "pose_detect"               : True,
            "pose_detect_thresh"        : 0.1,
            "person_detect"             : True,
            "person_detect_thresh"      : 0.30,
            "auto_label"                : False,
            "draw_labels_in_txt_file"   : True,     #   draw label(id,x,y,w,h) from txtfile
        }
    elif DEBUG == 1:
        # show label data
        config = {
            "show_image"                : True,
            "show_image_wait"           : 1.0,
            "dirs"                      : image_dir,
            "dirs_subsample_max"        : 1000,
            "face_detect"               : False,
            "face_detect_thresh"        : 0.2,
            "pose_detect"               : True,
            "pose_detect_thresh"        : 0.2,
            "person_detect"             : True,
            "person_detect_thresh"      : 0.30,
            "auto_label"                : False,
            "draw_labels_in_txt_file"   : False,     #   draw label(id,x,y,w,h) from txtfile
        }
    images = get_all_image_in_dir(config["dirs"], config["dirs_subsample_max"])
    auto_label_face_for_yolo(images, config)
