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


def show_image(labels_files, labels_prefix="labels_coco"):
    img_seq = 0
    running = True

    while running:
        if img_seq < 0: img_seq = 0
        if img_seq >= len(labels_files): break

        labelf = labels_files[img_seq]
        imagef = labelf.replace(labels_prefix, "images").replace(".txt", ".jpg")
        #print(imagef)
        frame = cv2.imread(imagef)
        h_ori, w_ori, _  = frame.shape
        IMG_SIZE = 2*416
        frame = cv2.resize(frame, (IMG_SIZE,IMG_SIZE))
        height, width, _  = frame.shape

        ff = open(labelf)
        yolo_labels = ff.readlines()
        ff.close()

        #show coco yoloface_fast_predictions
        if False:
            #"/home/leo/coco/images/val2017/000000036936.jpg"
            #"/home/leo/coco/yoloface_fast_predictions/val2017/xxxx.csv"
            cocoface_label = imagef.replace("images", "yoloface_fast_predictions") + ".csv"
            facesline = comm.read_filelist(cocoface_label)
            facebox = []
            for face in facesline:
                items = face.split(",")
                if (len(items) != 6):
                    continue
                #print(items[1:])
                #'/data/coco/val2017/000000036936.jpg,0.9891075,104.2295,414.4043,148.12671,440.77728'
                ppath, prob, y0, x0, y1, x1 = items
                x0 = float(x0) / w_ori
                y0 = float(y0) / h_ori
                x1 = float(x1) / w_ori
                y1 = float(y1) / h_ori
                prob = float(prob)
                w = x1 - x0
                h = y1 - y0
                #if w < comm.YOLO_FACE_MIN_SIZE or h < comm.YOLO_FACE_MIN_SIZE: continue
                #if w < comm.YOLO_FACE_MIN_SIZE or h < comm.YOLO_FACE_MIN_SIZE: continue
                facebox.append([prob,x0,y0,x1,y1])

            if len(facebox) > 1:
                idx_used = set()
                for ii in range(0, len(facebox)-1):
                    box0 = facebox[ii][1:]
                    for jj in range(ii+1, len(facebox)):
                        box1 = facebox[jj][1:]
                        iou = calc_iou(box0, box1)
                        print(ii, jj, box0, box1, 'iou:', iou)

            for face in facebox:
                prob,x0,y0,x1,y1 = face

                w = x1-x0
                h = y1-y0
                a_box = [x0, y0, w, h]

                yolo_id = 1
                info = [yolo_id, comm.id_to_names(yolo_id), prob, a_box]

                print(info)
                comm.draw_info_on_image(frame, width, height, info, comm.CC_FACE, 1)

        #yolo_labels = []
        w_over_h = False

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

            wh_ratio = box_width / box_height
            if yolo_id == comm.YOLO_FACE_ID and wh_ratio > 1.1:
                print(imagef, "w>h", info, wh_ratio)
                w_over_h = True

            #print(info)
            if yolo_id == comm.YOLO_FACE_ID:
                cc = comm.CC_FACE
            comm.draw_info_on_image(frame, width, height, info, cc, 1)

        #debug
        if not w_over_h:
            img_seq+=1
            continue

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
    COCO_TYPE='train2017'
    LABELS_PREFIX = "labels_coco"
    coco_image_list = f"/home/leo/coco/coco_{COCO_TYPE}_filelists.txt"
    images = comf.read_list(coco_image_list)
    labels_files = []
    for img in images:
        label = img.replace("images", LABELS_PREFIX).replace(".jpg", ".txt")
        labels_files.append(label)

    show_image(labels_files, LABELS_PREFIX)
