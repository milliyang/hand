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
YOLO_IMAGE_SIZE = (416,416)

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
}

folder_to_id = {
    'call'          : YOLO_HUMAN_HAND_CALL,
    'fist'          : YOLO_HUMAN_HAND_FIST,
    'four'          : YOLO_HUMAN_HAND_FOUR,
    'like'          : YOLO_HUMAN_HAND_LIKE,
    'ok'            : YOLO_HUMAN_HAND_OK,
    'one'           : YOLO_HUMAN_HAND_ONE,
    'palm'          : YOLO_HUMAN_HAND_FIVE,
    'stop'          : YOLO_HUMAN_HAND_FIVE,
    'stop_inverted' : YOLO_HUMAN_HAND_FIVE,
    'peace'         : YOLO_HUMAN_HAND_TWO,
    'peace_inverted': YOLO_HUMAN_HAND_TWO,
    'three'         : YOLO_HUMAN_HAND_THREE,
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
            all_y.append(y0)
            all_y.append(y1)

    if len(all_x) > 0 and len(all_y) > 0 and num >= min_point:
        #x,y,w,h
        x = min(all_x)
        y = min(all_y)
        w = max(all_x)-x
        h = max(all_y)-y
        info = (id, id_names[id], 1.0, (x,y,w,h))
        return info
    else:
        return None

def body_info_from_landmark(landmarks, hotfix_y_max = 1.0):
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
    #head        = get_rect_from_landmarks(YOLO_HUMAN_FACE_ID, landmarks, head_face_list, head_margin)
    body        = get_rect_from_landmarks(YOLO_HUMAN_BODY_ID, landmarks, body_marks_list, 0, 4, hotfix_y_max)
    #hand_left   = get_rect_from_landmarks(YOLO_HUMAN_HAND_ID, landmarks, hand_left_list,  0)
    #hand_right  = get_rect_from_landmarks(YOLO_HUMAN_HAND_ID, landmarks, hand_right_list, 0)
    #foot_left   = get_rect_from_landmarks(YOLO_HUMAN_FOOT_ID, landmarks, foot_left_list,  foot_margin)
    #foot_right  = get_rect_from_landmarks(YOLO_HUMAN_FOOT_ID, landmarks, foot_right_list, foot_margin)
    #
    leg  = get_rect_from_landmarks(YOLO_HUMAN_FOOT_ID, landmarks, leg_list, foot_margin, 5, hotfix_y_max)
    rects = []
    #if head: rects.append(head)                #ignore head
    #if hand_left: rects.append(hand_left)
    #if hand_right: rects.append(hand_right)
    #if foot_left: rects.append(foot_left)
    #if foot_right: rects.append(foot_right)
    if body: rects.append(body)
    if leg: rects.append(leg)
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

def info_to_yolo_string(info:list):
    yolo_id_idx, _, _, xywh = info
    #output yolo bbox fmt:  x,y,w,h -> cx,xy,w,h
    yolo_cxcywh = (xywh[0]+xywh[2]/2.0, xywh[1]+xywh[3]/2.0, xywh[2], xywh[3])
    yolo_string = f"{yolo_id_idx} {yolo_cxcywh[0]} {yolo_cxcywh[1]} {yolo_cxcywh[2]} {yolo_cxcywh[3]} \n"
    return yolo_string

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
            hand_gesture_label = []

            hotfix_one_person_y_max = 1.0

            face_box   = None #[x,y,w,h], 0~1.0
            person_box = None #[x,y,w,h], 0~1.0
            person_num = 0

            if config["person_detect"]:
                #print(dir(mp.Image))
                #imageL = mp.Image.create_from_file(imagef)
                #detection_result = object_detection.detect(imageL)
                rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                detection_result = object_detection.detect(rgb_frame)
                y_max = 1.0
                for detection in detection_result.detections:
                    category = detection.categories[0]
                    category_name = category.category_name
                    if category_name != 'person':
                        continue
                    pson_cc = (250, 100, 0)
                    # Draw bounding_box
                    bbox = detection.bounding_box           #fuck: 0~width
                    person_box = (bbox.origin_x/float(width), bbox.origin_y/float(height), bbox.width/float(width), bbox.height/float(height))
                    y_max = person_box[1] + person_box[3]
                    prob = round(category.score, 2)
                    info = [YOLO_HUMAN_HUMAN_ID, id_to_names(YOLO_HUMAN_HUMAN_ID), prob, person_box]
                    if config["show_image"]:
                        draw_info_on_image(image, width, height, info, pson_cc, 1)
                    person_label.append(info_to_yolo_string(info))
                    person_num+=1

                if person_num == 1:
                    hotfix_one_person_y_max = y_max

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
                    body_info = body_info_from_landmark(pose_result.pose_landmarks.landmark, hotfix_one_person_y_max)
                    body_cc = (250, 250, 0)
                    for a_info in body_info:
                        if config["show_image"]:
                            draw_info_on_image(image, width, height, a_info, body_cc, 1)
                        bodys_label.append(info_to_yolo_string(a_info))

            if config['face_detect']:
                face_cc = (0, 0, 250)
                if (face_results.detections):
                    for _, detection in enumerate(face_results.detections):
                        # mpDraw.draw_detection(image, detection)   #built-in function
                        # The box around the face
                        box = detection.location_data.relative_bounding_box
                        face_box = [box.xmin, box.ymin, box.width, box.height]

                        info = [YOLO_HUMAN_FACE_ID, id_to_names(YOLO_HUMAN_FACE_ID), detection.score[0], face_box]
                        if config["show_image"]:
                            draw_info_on_image(image, width, height, info, face_cc, 1)
                        faces_label.append(info_to_yolo_string(info))

            hagrid_labels = []
            #print(imagef)
            #image: /home/leo/myhome/hagrid/download/subsample/train/xxx/                   //xxx - guesture
            #label: /home/leo/myhome/hagrid/download/subsample/train_labels/xxx/
            labelfile = imagef.replace("train", "train_labels").replace(".jpg", ".txt")
            if config['hagrid_parse_labels']:
                ff = open(labelfile)
                hagrid_labels = ff.readlines()
                ff.close()

                # Leo:
                #  1. convert hand -> to number and hand
                #  2. if two hand found, the hand closer to face use number (because we check all the sample image)
                hands_box = []
                hand_gesture_box = None
                txtfile_cc = (0, 255, 120)

                for yolo_fmt in hagrid_labels:
                    items = yolo_fmt.strip().split()
                    yolo_id  = int(items[0])
                    #['0', '0.45230302', '0.2694478', '0.05382926', '0.11273142']

                    #cxcywh:
                    # python.exe  hagrid_to_yolo.py --bbox_format cxcywh
                    box_xmin   = float(items[1]) - float(items[3]) / 2.0
                    box_ymin   = float(items[2]) - float(items[4]) / 2.0
                    box_width  = float(items[3])
                    box_height = float(items[4])
                    #
                    a_box = [box_xmin, box_ymin, box_width, box_height]
                    info = [yolo_id, id_to_names(yolo_id), 1.0, a_box]
                    draw_info_on_image(image, width, height, info, txtfile_cc, 1)

                    if yolo_id == YOLO_HUMAN_HAND_ID:
                        hands_box.append(a_box)

                #
                if len(hands_box) > 0:
                    hand_gesture_box = hands_box[0]
                    if face_box != None:
                        min_y = hand_gesture_box[1]
                        for abox in hands_box:
                            if abox[1] < min_y:
                                min_y = abox[1]
                                hand_gesture_box = abox

                    # Generate Gesture
                    #  folder -> gesture
                    #    N:\hand_fullset\train\call\xxxx.jpg  -> 'call'  -> YOLO_HUMAN_HAND_CALL
                    #    N:\hand_fullset\train\three\xxxx.jpg -> 'three' -> YOLO_HUMAN_HAND_THREE
                    pathname = os.path.dirname(imagef)
                    basename = os.path.basename(pathname)
                    #print(imagef, pathname, basename)

                    if basename in folder_to_id.keys():
                        yolo_id = folder_to_id[basename]
                        info = [yolo_id, id_to_names(yolo_id), 1.0, hand_gesture_box]
                        if config["show_image"]:
                            draw_info_on_image(image, width, height, info, txtfile_cc, 2)
                        hand_gesture_label.append(info_to_yolo_string(info))

            if config["auto_label"]:
                if config['hagrid_parse_labels'] and person_num <= 0:
                    print(new_labelfile, "[skip][hagrid, no human]")
                else:
                    new_labelfile = labelfile.replace(".txt", "_mp_hand.txt")
                    file = open(new_labelfile, "w")
                    for each in hagrid_labels:
                        file.write(each)
                    for each in faces_label:
                        file.write(each)
                    for each in bodys_label:
                        file.write(each)
                    for each in person_label:
                        file.write(each)
                    for each in hand_gesture_label:
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
        "pose_detect_thresh"        : 0.2,
        "person_detect"             : True,
        "person_detect_thresh"      : 0.30,
        "auto_label"                : True,     #  xxxx.jpg -> xxxx._mp_hand.txt
        "hagrid_parse_labels"       : True,     # hagrid read hand label data
        "hagrid_must_has_person"    : True,     # hagrid image must has person; otherwise don't generate label files
    }

    DEBUG = 55555555
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
            "hagrid_parse_labels"       : True,     # hagrid read hand label data
            "hagrid_must_has_person"    : True,     # hagrid image must has person; otherwise don't generate label files
        }
    elif DEBUG == 1:
        # show label data
        config = {
            "show_image"                : True,
            "show_image_wait"           : 1.0,
            "dirs"                      : image_dir,
            "dirs_subsample_max"        : 5,
            "face_detect"               : True,
            "face_detect_thresh"        : 0.2,
            "pose_detect"               : True,
            "pose_detect_thresh"        : 0.2,
            "person_detect"             : True,
            "person_detect_thresh"      : 0.30,
            "auto_label"                : True,
            "hagrid_parse_labels"       : True,     # hagrid read hand label data
            "hagrid_must_has_person"    : True,     # hagrid image must has person; otherwise don't generate label files
        }
    images = get_all_image_in_dir(config["dirs"], config["dirs_subsample_max"])
    auto_label_face_for_yolo(images, config)
