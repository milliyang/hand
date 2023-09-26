import cv2
import mediapipe as mp
import os, time
import com_detection as comm
import com_files as comf
import sys

if __name__ == '__main__':
    input_file = "./output/sel_hagrid_filelist.txt"
    output_file = "./output/exclude_hagrid_filelist.txt"

    img_files = comf.read_filelist(input_file)
    txtfile_cc = (0, 255, 120)

    running = True
    image_files_exclude = []
    img_seq = 0
    if len(sys.argv) >= 2:
        img_seq = int(sys.argv[1])

    while running and img_seq < len(img_files):
        img_seq = max(img_seq, 0)

        imagef = img_files[img_seq]
        #for imagef in img_files:
        #print(imagef)
        frame = cv2.imread(imagef)
        IMAGE_SIZE = (416*2,416*2)
        #IMAGE_SIZE = comm.YOLO_IMAGE_SIZE
        frame = cv2.resize(frame, IMAGE_SIZE)
        height, width, _  = frame.shape

        #/mnt/214cd1a6-9d1b-4b2a-9770-425b64e6884f/myhome/dataset/VOCdevkit/VOC2012/JPEGImages/2008_007610.jpg
        #/mnt/214cd1a6-9d1b-4b2a-9770-425b64e6884f/myhome/dataset/VOCdevkit/VOC2012/labels/2008_007610.txt
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

            #if yolo_id == comm.YOLO_HAND_ID: print("hand", a_box)

            #print(info)
            comm.draw_seq_on_image(frame, img_seq)
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
            elif val == ord('d'):
                print("++", imagef, '[NG]')
                img_seq+=1
                image_files_exclude.append(imagef)
                break

    ss = set(image_files_exclude)
    image_files_exclude = list(image_files_exclude)
    comf.write_list(output_file, image_files_exclude, op = "a")

