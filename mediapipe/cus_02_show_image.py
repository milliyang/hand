import cv2
import mediapipe as mp
import os, time
import com_detection as comm
import com_files as comf

def calc_iou(box1, box2):
    """
    :param box1: = [xmin1, ymin1, xmax1, ymax1]
    :param box2: = [xmin2, ymin2, xmax2, ymax2]
    :return: 
    """
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    #
    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)  # b1 area
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)  # b2 area
 
    #
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)
 
    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)
    a1 = w * h  # C∩G
    a2 = s1 + s2 - a1
    iou = a1 / a2 #iou = a1/ (s1 + s2 - a1)
    return iou

def show_image(labels_files, labels_prefix="labels"):
    img_seq = 0
    running = True

    while running:
        if img_seq < 0: img_seq = 0
        if img_seq >= len(labels_files): break

        labelf = labels_files[img_seq]
        imagef = labelf.replace(labels_prefix, "images").replace(".txt", ".JPG")
        print(imagef)
        print(labelf)
        frame = cv2.imread(imagef)
        h_ori, w_ori, _  = frame.shape
        #IMG_SIZE = 2*416
        #frame = cv2.resize(frame, (IMG_SIZE,IMG_SIZE))
        height, width, _  = frame.shape

        ff = open(labelf)
        yolo_labels = ff.readlines()
        ff.close()

        #yolo_labels = []

        # Leo:
        #  1. convert hand -> to number and hand
        #  2. if two hand found, the hand closer to face use number (because we check all the sample image)
        cc = comm.CC_TXTFILE
        for yolo_fmt in yolo_labels:
            items = yolo_fmt.strip().split()
            if len(items) <= 0:
                continue
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
            #print(info)
            if yolo_id == comm.YOLO_FACE_ID:
                cc = comm.CC_FACE
            elif yolo_id == comm.YOLO_HUMAN_ID:
                cc = comm.CC_HUMAN
            comm.draw_info_on_image(frame, width, height, info, cc, 1)

        cv2.imshow('TrainImage', frame)

        while True:
            val = cv2.waitKey(1000) & 0xFF
            if val == ord(comm.EXIT_KEY_Q):
                running = False
                break
            elif val == ord('n') or val == 83:  #right
                print("++", imagef)
                img_seq+=1
                break
            elif val == ord('p') or val == 81:  #left
                print("--", imagef)
                img_seq-=1
                break


if __name__ == '__main__':
    LABELS_PREFIX = "labels"
    PATH_LABELS = "/home/leo/custom_face/labels"

    labels_files = comf.get_files_in_current_dir(PATH_LABELS)
    show_image(labels_files, LABELS_PREFIX)
