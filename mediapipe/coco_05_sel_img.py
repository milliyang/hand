import cv2
import mediapipe as mp
import os, time
import com_detection as comm
import com_files as comf
import sys

def select_image_files(images_files, output_file, img_seq = 0):
    txtfile_cc = (0, 255, 120)
    running = True
    image_files_exclude = []

    while running and img_seq < len(images_files):
        img_seq = max(img_seq, 0)
        imagef = images_files[img_seq]
        frame = cv2.imread(imagef)
        IMAGE_SIZE = (416*2,416*2)
        #IMAGE_SIZE = comm.YOLO_IMAGE_SIZE
        frame = cv2.resize(frame, IMAGE_SIZE)
        height, width, _  = frame.shape

        #/home/leo/coco/images/train2017/000000300024.jpg
        #/home/leo/coco/labels/train2017/000000300024.txt
        labelfile = imagef.replace(".jpg", ".txt").replace("images", "labels")

        if not os.path.isfile(labelfile):
            print(labelfile, '[no file][skip]')
            img_seq+=1
            continue

        ff = open(labelfile)
        yolo_labels = ff.readlines()
        ff.close()

        # Leo:
        #  1. convert hand -> to number and hand
        #  2. if two hand found, the hand closer to face use number (because we check all the sample image)
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
            info = [yolo_id, comm.id_to_names(yolo_id), 1.0, a_box]

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

if __name__ == '__main__':
    #DATA_TYPE = "train2017"
    DATA_TYPE = "val2017"
    input_file = f"./output/sel_{DATA_TYPE}_coco_filelist.txt"
    output_file = "./output/exclude_coco_filelist.txt"

    #input_file = f"./output/exclude_coco_filelist.txt"

    img_seq = 0
    if len(sys.argv) >= 2:
        img_seq = int(sys.argv[1])

    images_files = comf.read_filelist(input_file)

    select_image_files(images_files, output_file, img_seq)