import cv2
import mediapipe as mp
import os, time

EXIT_KEY_Q = 'q'

linefont = cv2.FONT_HERSHEY_SIMPLEX
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
        select_files.append(os.path.join(dir, onefile))
    return select_files

def get_all_files_in_dir(dir_list =[]):
    images = []
    for one_dir in dir_list:
        for entry in os.listdir(one_dir):
            path = os.path.join(one_dir, entry)
            if os.path.isdir(path):
                imgs = get_images_in_current_dir(path)
                images.extend(imgs)
                print(len(imgs), "totals:", len(images), path)
    return images


def select_file_with_pattern(files:list, pattern = ["mp_hand"]):
    select_files = []
    for each in files:
        for pp in pattern:
            if pp in each:
                select_files.append(each)
    return select_files

def draw_info_on_image(image, width, height, info:list, cc=(250, 250, 0), thinkness=1):
    _, yolo_name, prob, rect_xywh = info
    a_xywh = (rect_xywh[0]*width, rect_xywh[1]*height, rect_xywh[2]*width, rect_xywh[3]*height)
    a_xywh = int(a_xywh[0]), int(a_xywh[1]), int(a_xywh[2]), int(a_xywh[3])
    text = f'{yolo_name}:{prob:.2f}'
    #
    cv2.rectangle(image, a_xywh, cc, thinkness)
    cv2.putText(image, text, (a_xywh[0]+10, a_xywh[1]-10), linefont, 0.5, cc, thinkness)

def read_filelist(filelist):
    #filelist = "/home/leo/myhome/dataset/selected_voc_train.txt"
    afile = open(filelist)
    images = afile.readlines()
    images = [ima.strip() for ima in images]
    return images

if __name__ == '__main__':

    img_files = read_filelist("sel_voc_filelist.txt")

    for imagef in img_files:
        print(imagef)
        frame = cv2.imread(imagef)
        #frame = cv2.resize(frame, YOLO_IMAGE_SIZE)
        height, width, _  = frame.shape

        #/home/leo/hand_sample/train/three/714c1805-2402-4b35-94a1-162aad6d066c.jpg
        #/home/leo/hand_sample/train/three/714c1805-2402-4b35-94a1-162aad6d066c.txt
        labelfile = imagef.replace(".jpg", ".txt")

        labelfile = labelfile.replace("JPEGImages", "labels")

        print(labelfile)

        ff = open(labelfile)
        yolo_labels = ff.readlines()
        ff.close()

        # Leo:
        #  1. convert hand -> to number and hand
        #  2. if two hand found, the hand closer to face use number (because we check all the sample image)
        txtfile_cc = (0, 255, 120)
        

        for yolo_fmt in yolo_labels:
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

            if yolo_id == YOLO_HUMAN_HAND_ID:
                continue

            
            print(info)
            draw_info_on_image(frame, width, height, info, txtfile_cc, 1)

        cv2.imshow('TrainImage', frame)
        if (cv2.waitKey(1000*1) & 0xFF == ord(EXIT_KEY_Q)):
            break
