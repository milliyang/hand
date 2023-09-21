import cv2
import mediapipe as mp
import os, time
import com_detection as comm
import com_files as comf

if __name__ == '__main__':

    coco_image_list = "/home/leo/coco/coco_val2017_filelists.txt"
    images = comf.read_list(coco_image_list)
    labels_files = []
    for img in images:
        label = img.replace("images", "labels_coco").replace(".jpg", ".txt")
        labels_files.append(label)

    img_seq = 0
    running = True

    while running:
        if img_seq < 0: img_seq = 0
        if img_seq >= len(labels_files): break

        labelf = labels_files[img_seq]
        imagef = labelf.replace("labels_coco", "images").replace(".txt", ".jpg")
        print(imagef)
        frame = cv2.imread(imagef)
        h_ori, w_ori, _  = frame.shape
        IMG_SIZE = 2*416
        frame = cv2.resize(frame, (IMG_SIZE,IMG_SIZE))
        height, width, _  = frame.shape
  
        ff = open(labelf)
        yolo_labels = ff.readlines()
        ff.close()

        #show coco yoloface_fast_predictions
        if True:
            #"/home/leo/coco/images/val2017/000000036936.jpg"
            #"/home/leo/coco/yoloface_fast_predictions/val2017/xxxx.csv"
            cocoface_label = imagef.replace("images", "yoloface_fast_predictions") + ".csv"
            faces = comm.read_filelist(cocoface_label)
            face_cc = (250, 0, 0)
            for face in faces:
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
                w = x1-x0
                h = y1-y0
                xywh = [x0, y0, x1-x0, y1-y0]
                #rect_xywh = [int(xywh[0]*IMG_SIZE), int(xywh[1]*IMG_SIZE), int(xywh[2]*IMG_SIZE), int(xywh[3]*IMG_SIZE)]
                #cv2.rectangle(frame, rect_xywh, face_cc, 1)

                yolo_id = 1
                a_box = xywh
                info = [yolo_id, comm.id_to_names(yolo_id), float(prob), a_box]
                if a_box[2] < 0.02 or a_box[2] < 0.02:
                    continue

                print(info)
                comm.draw_info_on_image(frame, width, height, info, face_cc, 1)

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