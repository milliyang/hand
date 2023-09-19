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

def read_list(filename):
    #filename = "/home/leo/myhome/dataset/selected_voc_train.txt"
    afile = open(filename)
    images = afile.readlines()
    images = [ima.strip() for ima in images]
    return images

def write_list(filename, alist: list, op = "w"):
    file = open(filename, op)
    for item in alist:
        file.write(item)
        file.write("\n")
    file.close()

def write_list_to_file(alist: list, filename="filelist.txt"):
    write_list(filename, alist)
    print("generate:", filename, "  num:", len(alist))

def read_filelist(filename):
    return read_list(filename)
