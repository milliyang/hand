import cv2
import mediapipe as mp
import os, time

import com_detection as comm
import com_files as comf

def auto_label_vol_for_yolo(imagefiles = [], config = {}):
    mpFaceDetector = mp.solutions.face_detection
    mpDraw = mp.solutions.drawing_utils
    mp_face_cfg = {
        "min_detection_confidence" : config['face_detect_thresh'], #0.5
        "model_selection" : 1,      # 1,near,far; 0,near;
    }
    pose_detection   = comm.post_get_detector(config['pose_detect_thresh'])
    object_detection = comm.get_object_detector(config['person_detect_thresh'])
    hand_detection   = comm.get_hand_detector(config['hand_detect_thresh'])

    image_seq = config['image_seq_number']
    image_seq_max = len(imagefiles)-1

    #cap = cv2.VideoCapture(0)
    with mpFaceDetector.FaceDetection(**mp_face_cfg) as faceDetection:

        #for image_seq, imagef in enumerate(imagefiles):
        while image_seq < image_seq_max:
            #print(imagef)
            imagef = imagefiles[image_seq]
            image_seq+=1

            frame = cv2.imread(imagef)
            frame = cv2.resize(frame, comm.YOLO_IMAGE_SIZE)
            height, width, _  = frame.shape

            # cv2 uses BGR and mediapipe uses RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert it back for displaying after processing
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            faces_label = []
            bodys_label = []
            person_label = []
            hand_label = []

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

            voc_labels = []
            voc_valid_labels = []

            #/home/leo/coco/images/train2017/000000300024.jpg
            #/home/leo/coco/labels_coco/train2017/000000300024.txt
            #or
            #/home/leo/coco/images/val2017/000000300024.jpg
            #/home/leo/coco/labels_coco/val2017/000000300024.txt
            labelfile = imagef.replace("images", "labels_coco").replace(".jpg", ".txt")
            coco_has_person = 0

            if config['coco_parse_labels']:
                ff = open(labelfile)
                voc_labels = ff.readlines()
                ff.close()
                #
                txtfile_cc = (0, 255, 120)
                for yolo_fmt in voc_labels:
                    items = yolo_fmt.strip().split()
                    if len(items) <= 0:
                        continue
                    yolo_id = int(items[0])
                    #['0', '0.45230302', '0.2694478', '0.05382926', '0.11273142']
                    #cxcywh:
                    # python.exe  hagrid_to_yolo.py --bbox_format cxcywh
                    box_xmin   = float(items[1]) - float(items[3]) / 2.0
                    box_ymin   = float(items[2]) - float(items[4]) / 2.0
                    box_width  = float(items[3])
                    box_height = float(items[4])

                    #ignore face too small
                    if box_height < comm.YOLO_OBJECT_MIN_SIZE or box_width < comm.YOLO_OBJECT_MIN_SIZE: continue

                    if yolo_id == comm.YOLO_HUMAN_ID:
                        coco_has_person+=1

                    a_box = [box_xmin, box_ymin, box_width, box_height]
                    info  = [yolo_id, comm.id_to_names(yolo_id), 1.0, a_box]
                    #print(info)

                    comm.draw_info_on_image(image, width, height, info, txtfile_cc, 1)
                    voc_valid_labels.append(comm.info_to_yolo_string(info))
                #

            if coco_has_person:
                if config["pose_detect"]:
                    pose_result = pose_detection.process(image_rgb)
                    #comm.pose_draw_pose_landmarks(image, pose_result.pose_landmarks)

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
                    face_results = faceDetection.process(image_rgb)
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

                if config['hand_detect']:
                    hand_cc = (30, 30, 200)
                    results = hand_detection.process(image_rgb)
                    landmarks = results.multi_hand_landmarks
                    if landmarks:
                        #comm.draw_hand_landmarks(image, landmarks)
                        infos = comm.hand_info_from_landmark(landmarks)
                        for info in infos:
                            hand_label.append(comm.info_to_yolo_string(info))
                            if config["show_image"]:
                                comm.draw_info_on_image(image, width, height, info, hand_cc, 1)

            if config["auto_label"]:
                new_labelfile = imagef.replace("images", "labels").replace(".jpg", ".txt")
                if config['coco_parse_labels'] and coco_has_person <= 0:
                    print(new_labelfile, "[skip][no human]")
                else:
                    labels = []
                    labels.extend(voc_valid_labels)
                    labels.extend(faces_label)
                    labels.extend(bodys_label)
                    labels.extend(person_label)
                    labels.extend(hand_label)
                    if len(labels) > 0:
                        comf.ensure_file_dir(new_labelfile)
                        comf.write_list_to_file(labels, new_labelfile)

            if config["show_image"]:
                cv2.imshow('Coco detection', image)
                wait_time = config["show_image_wait"]
                if (cv2.waitKey(10) & 0xFF == ord(comm.EXIT_KEY)):
                    break
                if wait_time > 0:
                    time.sleep(wait_time)

            if image_seq > config["max_image_seq_number"]:
                print("image_seq:", image_seq, " finish [TOO MUCH IMAGE]")
                break

if __name__ == '__main__':
    FACE_THRESH = 0.75
    POSE_THRESH = 0.90
    HAND_THRESH = 0.80

    config = {
        "show_image"                : False,
        "show_image_wait"           : 0,
        "face_detect"               : False,        # no need to detect face, we use COCO.face.yolo.project labels
        "face_detect_thresh"        : FACE_THRESH,
        "pose_detect"               : True,
        "pose_detect_thresh"        : POSE_THRESH,
        "person_detect"             : False,        # no need to detect person, VOC/COCO already has person
        "person_detect_thresh"      : 0.80,
        "hand_detect"               : True,
        "hand_detect_thresh"        : HAND_THRESH,
        "auto_label"                : True,     #  xxxx.jpg -> xxxx._mp_hand.txt
        "coco_parse_labels"         : True,     # hagrid read hand label data
        "image_seq_number"          : 0,
        "max_image_seq_number"      : 30000,
    }

    DEBUG = 0
    if DEBUG == 1:
        config["show_image"]            = True
        config["show_image_wait"]       = 0.5
        config["auto_label"]            = False

    #coco_image_list = "/home/leo/coco/coco_filelists.txt"
    COCO_TYPE = "train2017"
    #COCO_TYPE = "val2017"

    coco_image_list = f"/home/leo/coco/coco_{COCO_TYPE}_filelists.txt"
    images = comf.read_list(coco_image_list)

    #clear dir
    #old_labels = f"/home/leo/coco/labels/{COCO_TYPE}"
    #comf.remove_all_files_in_cur_dir(old_labels)

    auto_label_vol_for_yolo(images, config)
