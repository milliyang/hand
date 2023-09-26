import cv2
import os, time

import com_detection as comm
import com_files as comf

if __name__ == '__main__':
    path  = "/home/leo/hand_sample/train"

    all_files = comf.get_all_files_in_dir([path])
    img_files = comf.select_file_with_pattern(all_files, ['jpg', 'png'])
    txtfile_cc = (0, 255, 120)

    for imagef in img_files:
        frame = cv2.imread(imagef)
        frame = cv2.resize(frame, comm.YOLO_IMAGE_SIZE)
        height, width, _  = frame.shape

        #/home/leo/hand_sample/train/three/714c1805-2402-4b35-94a1-162aad6d066c.jpg
        #/home/leo/hand_sample/train/three/714c1805-2402-4b35-94a1-162aad6d066c.txt
        labelfile = imagef.replace(".jpg", ".txt")
        #print(labelfile)

        ff = open(labelfile)
        yolo_labels = ff.readlines()
        ff.close()

        # Leo:
        #  1. convert hand -> to number and hand
        #  2. if two hand found, the hand closer to face use number (because we check all the sample image)
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

            #in hagrid: hand and hand_gesture overlay
            if yolo_id == comm.YOLO_HAND_ID:
                continue

            #print(info)
            comm.draw_info_on_image(frame, width, height, info, txtfile_cc, 1)

        cv2.imshow('TrainImage', frame)
        if (cv2.waitKey(1000*1) & 0xFF == ord(comm.EXIT_KEY_Q)):
            break
