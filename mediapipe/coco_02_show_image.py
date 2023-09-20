import cv2
import mediapipe as mp
import os, time
import com_detection as comm
import com_files as comf

if __name__ == '__main__':

    #COCO_PATH = "/home/leo/coco/labels_coco/train2017"
    COCO_PATH = "/home/leo/coco/labels_coco/val2017"

    #labels_files = comf.get_all_files_in_dir([COCO_PATH])
    labels_files = comf.get_files_in_current_dir(COCO_PATH)

    img_seq = 0
    running = True

    while running:
        if img_seq < 0: img_seq = 0
        if img_seq >= len(labels_files): break

        labelf = labels_files[img_seq]
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
            print(info)

            comm.draw_info_on_image(frame, width, height, info, txtfile_cc, 1)

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