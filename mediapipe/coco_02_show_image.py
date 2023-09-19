import cv2
import mediapipe as mp
import os, time
import com_detection as comm
import com_files as comf

if __name__ == '__main__':

    COCO_PATH = "/home/leo/coco/labels_coco"

    labels_files = comf.get_all_files_in_dir([COCO_PATH])

    for labelf in labels_files:
        imagef = labelf.replace("labels_coco", "images").replace(".txt", ".jpg")
        print(imagef)
        frame = cv2.imread(imagef)
        frame = cv2.resize(frame, comm.YOLO_IMAGE_SIZE)
        height, width, _  = frame.shape
  
        ff = open(labelf)
        yolo_labels = ff.readlines()
        ff.close()

        # Leo:
        #  1. convert hand -> to number and hand
        #  2. if two hand found, the hand closer to face use number (because we check all the sample image)
        txtfile_cc = (0, 255, 120)
        
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

            comm.draw_info_on_image(frame, width, height, info, txtfile_cc, 1)

        cv2.imshow('TrainImage', frame)

        if (cv2.waitKey(1000*10) & 0xFF == ord(comm.EXIT_KEY_Q)):
            break