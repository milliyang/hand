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
    pose_detection = comm.post_get_detector(config['pose_detect_thresh'])
    object_detection = comm.get_object_detector(config['person_detect_thresh'])
    hand_detection = comm.get_hand_detector(config['hand_detect_thresh'])

    #cap = cv2.VideoCapture(0)
    with mpFaceDetector.FaceDetection(**mp_face_cfg) as faceDetection:

        for _, imagef in enumerate(imagefiles):
            #print(imagef)

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
            #print(imagef)
            #image: /home/leo/myhome/dataset/VOCdevkit/VOC2012/JPEGImages/
            #label: /home/leo/myhome/dataset/VOCdevkit/VOC2012/labels_voc/
            labelfile = imagef.replace("JPEGImages", "labels_voc").replace(".jpg", ".txt")

            voc_has_person = 0

            if config['voc_parse_labels']:
                ff = open(labelfile)
                voc_labels = ff.readlines()
                ff.close()
                #
                txtfile_cc = (0, 255, 120)
                for yolo_fmt in voc_labels:
                    items = yolo_fmt.strip().split()
                    voc_id  = int(items[0])
                    yolo_id = comm.voc_class_id_to_imvt_class_id(voc_id)
                    if yolo_id < 0:
                        continue

                    if yolo_id == comm.YOLO_HUMAN_ID:
                        voc_has_person+=1
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
                    voc_valid_labels.append(comm.info_to_yolo_string(info))   
                #

            if voc_has_person:
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
                new_labelfile = labelfile.replace(".txt", "_mp_hand.txt")

                labels = []
                labels.extend(voc_valid_labels)
                labels.extend(faces_label)
                labels.extend(bodys_label)
                labels.extend(person_label)
                labels.extend(hand_label)

                if len(labels) > 0:
                    file = open(new_labelfile, "w")
                    for each in labels:
                        file.write(each)
                    file.close()
                    print(new_labelfile)

            if config["show_image"]:
                cv2.imshow('Face detection', image)
                wait_time = config["show_image_wait"]
                if (cv2.waitKey(10) & 0xFF == ord(comm.EXIT_KEY)):
                    break
                if wait_time > 0:
                    time.sleep(wait_time)

if __name__ == '__main__':
    config = {
        "show_image"                : False,
        "show_image_wait"           : 0,
        "face_detect"               : True,
        "face_detect_thresh"        : 0.2,
        "pose_detect"               : True,
        "pose_detect_thresh"        : 0.2,
        "person_detect"             : False,    # no need to detect person, VOC already has person
        "person_detect_thresh"      : 0.30,
        "hand_detect"               : True,
        "hand_detect_thresh"        : 0.3,
        "auto_label"                : True,     #  xxxx.jpg -> xxxx._mp_hand.txt
        "voc_parse_labels"          : True,     # hagrid read hand label data
    }

    DEBUG = 1
    if DEBUG == 1:
        config["show_image"]        = True
        config["show_image_wait"]   = 1.0
        config["auto_label"]        = False

    vol_image_filelist = "/home/leo/myhome/dataset/selected_voc_train.txt"
    images = comf.read_filelist(vol_image_filelist)
    auto_label_vol_for_yolo(images, config)
