import cv2
import mediapipe as mp
import os, time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

EXIT_KEY_Q = 'q'
EXIT_KEY = 'q'

YOLO_OBJECT_MIN_SIZE = 0.03

file_ext = ['jpg', 'jpeg']
YOLO_IMAGE_SIZE = (416,416)

YOLO_HUMAN_ID   = 0
YOLO_FACE_ID    = 1
YOLO_HAND_ID    = 2
YOLO_BODY_ID    = 3
YOLO_FOOT_ID    = 4
#
YOLO_HAND_ONE   = 5
YOLO_HAND_TWO   = 6     #peace, peace_inv, two_up
YOLO_HAND_THREE = 7
YOLO_HAND_FOUR  = 8
YOLO_HAND_FIVE  = 9     #five,stop        -->Larger
YOLO_HAND_OK    = 10    #                 -->Smaller
YOLO_HAND_LIKE  = 11
YOLO_HAND_CALL  = 12
YOLO_HAND_FIST  = 13
#
YOLO_DOG_ID      = 14
YOLO_CAT_ID      = 15
YOLO_BIRD_ID     = 16
YOLO_HORSE_ID    = 17
YOLO_SHEEP_ID    = 18
YOLO_CAR_ID      = 19
YOLO_ID_NUM      = 20

IMVT_CLS_NAMES = [
    "person",
    "face",
    "hand",
    "body",
    "foot",
    #
    "one",
    "two",
    "three",
    "four",
    "five",
    "ok",
    "like",
    "call",
    "fist",
    #
    "dog",
    "cat",
    "bird",
    "horse",
    "sheep",
    "car",
]

folder_to_id = {
    'call'          : YOLO_HAND_CALL,
    'fist'          : YOLO_HAND_FIST,
    'four'          : YOLO_HAND_FOUR,
    'like'          : YOLO_HAND_LIKE,
    'ok'            : YOLO_HAND_OK,
    'one'           : YOLO_HAND_ONE,
    'palm'          : YOLO_HAND_FIVE,
    'stop'          : YOLO_HAND_FIVE,
    'stop_inverted' : YOLO_HAND_FIVE,
    'peace'         : YOLO_HAND_TWO,
    'peace_inverted': YOLO_HAND_TWO,
    'three'         : YOLO_HAND_THREE,
}

VOC_CLASS_NAMES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

COCO_ID_NAME_MAP ={1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
                   6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
                   11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
                   16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
                   22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
                   28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
                   35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
                   40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket',
                   44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
                   51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
                   56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
                   61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
                   70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard',
                   77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
                   82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors',
                   88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}

def voc_class_id_to_imvt_class_id(id:int):
    name = VOC_CLASS_NAMES[id]
    if name in IMVT_CLS_NAMES:
        idx = IMVT_CLS_NAMES.index(name)
        return idx
    else:
        return -1

linefont = cv2.FONT_HERSHEY_SIMPLEX

def id_to_names(id):
    id = int(id)
    if id < YOLO_ID_NUM:
        return IMVT_CLS_NAMES[id]
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

def get_rect_from_landmarks(id, landmarks, alist:list, margin=0, min_point = 1, hotfix_y_max = 1.0):
    all_x = []
    all_y = []
    num = 0
    for mark in alist:
        x = landmarks[mark].x
        y = landmarks[mark].y
        if (x >= 0) and (x <= 1.0) and (y >= 0) and (y <= 1.0):
            num+=1

            x0 = min(x - margin, hotfix_y_max)
            x1 = min(x + margin, hotfix_y_max)
            x2 = min(x,          hotfix_y_max)
            x0 = max(x0, 0.0)
            all_x.append(x0)
            all_x.append(x1)
            all_x.append(x2)

            y0 = min(y - margin, hotfix_y_max)
            y1 = min(y + margin, hotfix_y_max)
            y2 = min(y,          hotfix_y_max)
            y0 = max(y0, 0.0)
            all_y.append(y0)
            all_y.append(y1)
            all_y.append(y2)

    if len(all_x) > 0 and len(all_y) > 0 and num >= min_point:
        #x,y,w,h
        x = min(all_x)
        y = min(all_y)
        w = max(all_x)-x
        h = max(all_y)-y
        info = (id, IMVT_CLS_NAMES[id], 1.0, (x,y,w,h))
        return info
    else:
        return None

def post_get_detector(thresh=0.5):
    mp_pose = mp.solutions.pose
    mp_pose_cfg = {
        "static_image_mode"         : True,
        "model_complexity"          : 0,    #0,1,2
        "enable_segmentation"       : False,
        "min_detection_confidence"  : thresh, #0.5
        "upper_body_only"           : False,
        "enable_segmentation"       : False,
        "smooth_segmentation"       : False,
        "min_tracking_confidence"   : thresh,
    }
    detector = mp_pose.Pose(mp_pose_cfg)
    return detector

def pose_draw_pose_landmarks(image, landmarks):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    mp_drawing.draw_landmarks(image, landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

def body_info_from_landmark(landmarks, hotfix_y_max = 1.0):
    mp_pose = mp.solutions.pose
    INDEX = mp_pose.PoseLandmark
    #margin_x = landmarks[INDEX.LEFT_EYE].x - landmarks[INDEX.RIGHT_EYE].x
    #margin_y = landmarks[INDEX.LEFT_EYE].y - landmarks[INDEX.RIGHT_EYE].y
    #head_margin = max(margin_x, margin_y)
    #print(margin)

    foot_margin_x = landmarks[INDEX.LEFT_HEEL].x - landmarks[INDEX.LEFT_FOOT_INDEX].x
    foot_margin_y = landmarks[INDEX.LEFT_HEEL].y - landmarks[INDEX.LEFT_FOOT_INDEX].y
    foot_margin = max(foot_margin_x, foot_margin_y) / 1.5

    #head_face_list   = [INDEX.NOSE, INDEX.LEFT_EYE, INDEX.RIGHT_EYE, INDEX.LEFT_EAR, INDEX.RIGHT_EAR, INDEX.MOUTH_LEFT, INDEX.MOUTH_RIGHT]
    #body_marks_list  = [INDEX.LEFT_SHOULDER, INDEX.RIGHT_SHOULDER, INDEX.LEFT_HIP, INDEX.RIGHT_HIP, INDEX.LEFT_KNEE, INDEX.RIGHT_KNEE]
    body_marks_list  = [INDEX.LEFT_SHOULDER, INDEX.RIGHT_SHOULDER, INDEX.LEFT_HIP, INDEX.RIGHT_HIP]
    #
    #hand_left_list   = [INDEX.LEFT_WRIST, INDEX.LEFT_PINKY, INDEX.LEFT_INDEX, INDEX.LEFT_THUMB]
    #hand_right_list  = [INDEX.RIGHT_WRIST, INDEX.RIGHT_PINKY, INDEX.RIGHT_INDEX, INDEX.RIGHT_THUMB]
    #foot_left_list   = [INDEX.LEFT_ANKLE, INDEX.LEFT_HEEL, INDEX.LEFT_FOOT_INDEX]
    #foot_right_list  = [INDEX.RIGHT_ANKLE, INDEX.RIGHT_HEEL, INDEX.RIGHT_FOOT_INDEX]
    #
    leg_list  = [INDEX.RIGHT_ANKLE, INDEX.RIGHT_HEEL, INDEX.RIGHT_FOOT_INDEX, INDEX.RIGHT_KNEE,
                 INDEX.LEFT_ANKLE,  INDEX.LEFT_HEEL,  INDEX.LEFT_FOOT_INDEX,  INDEX.LEFT_KNEE]
    #head        = get_rect_from_landmarks(YOLO_FACE_ID, landmarks, head_face_list, head_margin)
    body        = get_rect_from_landmarks(YOLO_BODY_ID, landmarks, body_marks_list, 0, 4, hotfix_y_max)
    #hand_left   = get_rect_from_landmarks(YOLO_HAND_ID, landmarks, hand_left_list,  0)
    #hand_right  = get_rect_from_landmarks(YOLO_HAND_ID, landmarks, hand_right_list, 0)
    #foot_left   = get_rect_from_landmarks(YOLO_FOOT_ID, landmarks, foot_left_list,  foot_margin)
    #foot_right  = get_rect_from_landmarks(YOLO_FOOT_ID, landmarks, foot_right_list, foot_margin)
    #
    leg  = get_rect_from_landmarks(YOLO_FOOT_ID, landmarks, leg_list, foot_margin, 5, hotfix_y_max)
    rects = []
    #if head: rects.append(head)                #ignore head
    #if hand_left: rects.append(hand_left)
    #if hand_right: rects.append(hand_right)
    #if foot_left: rects.append(foot_left)
    #if foot_right: rects.append(foot_right)
    if body:
        if body[3][2] > YOLO_OBJECT_MIN_SIZE and body[3][3] > YOLO_OBJECT_MIN_SIZE:
            rects.append(body)
    if leg:
        if leg[3][2] > YOLO_OBJECT_MIN_SIZE and leg[3][3] > YOLO_OBJECT_MIN_SIZE:
            rects.append(leg)
    #print(rects)
    return rects

def get_object_detector(thresh=0.40):
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    #
    base_options = python.BaseOptions(model_asset_path='efficientdet_lite2.tflite')
    options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=thresh)
    detector = vision.ObjectDetector.create_from_options(options)
    return detector

def get_hand_detector(thresh=0.40):
    mpHands = mp.solutions.hands
    detector = mpHands.Hands(min_detection_confidence=thresh, min_tracking_confidence=0.5, max_num_hands=4)
    return detector

def calc_rect_distance(rect1, rect2):
    ''' d^2 = x^2 + y^2 '''
    c1_x = rect1[0] + rect1[2] / 2.0
    c1_y = rect1[1] + rect1[3] / 2.0

    c2_x = rect2[0] + rect2[2] / 2.0
    c2_y = rect2[1] + rect2[3] / 2.0
    return pow(c1_x-c2_x, 2) + pow(c1_y-c2_y, 2)

def draw_info_on_image(image, width, height, info:list, cc=(250, 250, 0), thinkness=1):
    _, yolo_name, prob, rect_xywh = info
    a_xywh = (rect_xywh[0]*width, rect_xywh[1]*height, rect_xywh[2]*width, rect_xywh[3]*height)
    a_xywh = int(a_xywh[0]), int(a_xywh[1]), int(a_xywh[2]), int(a_xywh[3])
    text = f'{yolo_name}:{prob:.2f}'
    #
    cv2.rectangle(image, a_xywh, cc, thinkness)
    cv2.putText(image, text, (a_xywh[0]+10, a_xywh[1]-10), linefont, 0.5, cc, thinkness)

def draw_seq_on_image(image, seq:int, cc=(0, 0, 250), thinkness=2):
    text = f'{seq}'
    cv2.putText(image, text, (10, 20), linefont, 0.5, cc, thinkness)

def info_to_yolo_string(info:list):
    yolo_id_idx, _, _, xywh = info
    #output yolo bbox fmt:  x,y,w,h -> cx,xy,w,h
    yolo_cxcywh = (xywh[0]+xywh[2]/2.0, xywh[1]+xywh[3]/2.0, xywh[2], xywh[3])
    yolo_string = f"{yolo_id_idx} {yolo_cxcywh[0]} {yolo_cxcywh[1]} {yolo_cxcywh[2]} {yolo_cxcywh[3]} \n"
    return yolo_string

def hand_get_rect_from_landmarks(id, landmarks, margin=0.015, min_point = 1):
    all_x = []
    all_y = []
    num = 0
    #print(landmarks.landmark)
    #for each in landmarks.landmark:
    #    print('each', each.x, each.y, each.z)
    for mark in landmarks.landmark:
        x = mark.x
        y = mark.y
        z = mark.z
        if (x >= 0) and (x <= 1.0) and (y >= 0) and (y <= 1.0):
            num+=1
            x0 = min(x - margin, 1.0)
            x1 = min(x + margin, 1.0)
            x2 = min(x,          1.0)
            x0 = max(x0, 0.0)
            all_x.append(x0)
            all_x.append(x1)
            all_x.append(x2)
            y0 = min(y - margin, 1.0)
            y1 = min(y + margin, 1.0)
            y2 = min(y,          1.0)
            y0 = max(y0, 0.0)
            all_y.append(y0)
            all_y.append(y1)
            all_y.append(y2)

    if len(all_x) > 0 and len(all_y) > 0 and num >= min_point:
        #x,y,w,h
        x = min(all_x)
        y = min(all_y)
        w = max(all_x)-x
        h = max(all_y)-y
        info = (id, IMVT_CLS_NAMES[id], 1.0, (x,y,w,h))
        return info
    else:
        return None

def hand_info_from_landmark(landmarks):
    infos = []
    if landmarks:
        for hand in landmarks:
            info = hand_get_rect_from_landmarks(YOLO_HAND_ID, hand)
            if info:
                #print(info)
                infos.append(info)
    return infos

def draw_hand_landmarks(image, landmarks):
    if landmarks:
        mp_drawing = mp.solutions.drawing_utils
        mpHands = mp.solutions.hands
        DOT_COLOR = (0, 0, 255)
        CONNECTION_COLOR = (0, 255, 0)
        
        for hand in landmarks:
            mp_drawing.draw_landmarks(
                image, hand, mpHands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=DOT_COLOR, thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=CONNECTION_COLOR, thickness=2, circle_radius=2)
            )

def read_filelist(filelist):
    #filelist = "/home/leo/myhome/dataset/selected_voc_train.txt"
    afile = open(filelist)
    images = afile.readlines()
    images = [ima.strip() for ima in images]
    return images
