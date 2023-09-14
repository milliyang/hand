import cv2
import mediapipe as mp
import os, time

import com_detection as comm

def auto_label_face_for_yolo(imagefiles = [], config = {}):
    mpFaceDetector = mp.solutions.face_detection
    mpDraw = mp.solutions.drawing_utils
    mp_face_cfg = {
        "min_detection_confidence" : config['face_detect_thresh'], #0.5
        "model_selection" : 1,      # 1,near,far; 0,near;
    }

    pose_detection = comm.post_get_detector(config['pose_detect_thresh'])
    object_detection = comm.get_object_detector(config['person_detect_thresh'])

    #cap = cv2.VideoCapture(0)
    with mpFaceDetector.FaceDetection(**mp_face_cfg) as faceDetection:

        for _, imagef in enumerate(imagefiles):
            frame = cv2.imread(imagef)
            frame = cv2.resize(frame, comm.YOLO_IMAGE_SIZE)
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
                    info = [comm.YOLO_HUMAN_ID, comm.id_to_names(comm.YOLO_HUMAN_ID), prob, person_box]
                    if config["show_image"]:
                        comm.draw_info_on_image(image, width, height, info, pson_cc, 1)
                    person_label.append(comm.info_to_yolo_string(info))
                    person_num+=1

                if person_num == 1:
                    hotfix_one_person_y_max = y_max

            if config["pose_detect"]:
                pose_result = pose_detection.process(image_rgb)
                if config["show_image"]:
                    comm.pose_draw_pose_landmarks(image, pose_result.pose_landmarks)
                if pose_result.pose_landmarks :
                    #https://blog.csdn.net/weixin_43229348/article/details/120541448
                    #https://developers.google.cn/android/reference/com/google/mlkit/vision/pose/PoseLandmark
                    body_info = comm.body_info_from_landmark(pose_result.pose_landmarks.landmark, hotfix_one_person_y_max)
                    body_cc = (250, 250, 0)
                    for a_info in body_info:
                        if config["show_image"]:
                            comm.draw_info_on_image(image, width, height, a_info, body_cc, 1)
                        bodys_label.append(comm.info_to_yolo_string(a_info))

            if config['face_detect']:
                face_cc = (0, 0, 250)
                if (face_results.detections):
                    for _, detection in enumerate(face_results.detections):
                        # mpDraw.draw_detection(image, detection)   #built-in function
                        # The box around the face
                        box = detection.location_data.relative_bounding_box
                        face_box = [box.xmin, box.ymin, box.width, box.height]

                        info = [comm.YOLO_FACE_ID, comm.id_to_names(comm.YOLO_FACE_ID), detection.score[0], face_box]
                        if config["show_image"]:
                            comm.draw_info_on_image(image, width, height, info, face_cc, 1)
                        faces_label.append(comm.info_to_yolo_string(info))

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
                    info = [yolo_id, comm.id_to_names(yolo_id), 1.0, a_box]
                    comm.draw_info_on_image(image, width, height, info, txtfile_cc, 1)

                    if yolo_id == comm.YOLO_HAND_ID:
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
                    #    N:\hand_fullset\train\call\xxxx.jpg  -> 'call'  -> comm.YOLO_HAND_CALL
                    #    N:\hand_fullset\train\three\xxxx.jpg -> 'three' -> comm.YOLO_HAND_THREE
                    pathname = os.path.dirname(imagef)
                    basename = os.path.basename(pathname)
                    #print(imagef, pathname, basename)

                    if basename in comm.folder_to_id.keys():
                        yolo_id = comm.folder_to_id[basename]
                        info = [yolo_id, comm.id_to_names(yolo_id), 1.0, hand_gesture_box]
                        if config["show_image"]:
                            comm.draw_info_on_image(image, width, height, info, txtfile_cc, 2)
                        hand_gesture_label.append(comm.info_to_yolo_string(info))

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
                if (cv2.waitKey(10) & 0xFF == ord(comm.EXIT_KEY)):
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

    DEBUG = 1
    if DEBUG == 1:
        config["show_image"]                = True
        config["show_image_wait"]           = 1
        config["dirs_subsample_max"]        = 5
        config["auto_label"]                = False

    images = comm.get_all_image_in_dir(config["dirs"], config["dirs_subsample_max"])
    auto_label_face_for_yolo(images, config)
