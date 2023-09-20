import cv2
import mediapipe as mp
import os, time
import com_detection as comm
import com_files as comf

def copy_and_convert_cxcywh_xyxy_file(source, target):
    if os.path.isfile(target):
        os.remove(target)

    lines = comf.read_list(source)

    fmt_line = []
    for line in lines:
        items = line.split()

        new_items = []
        id = 0
        prob = 0
        bbox = None
        if len(items) == 5:
            id = items[0]
            bbox = items[1:]
        elif len(items) == 6:
            id   = items[0]
            prob = float(items[1])
            bbox = items[2:]
        #cx,cy,w,h => xy,xy
        left   = float(bbox[0]) - float(bbox[2])/2.0
        top    = float(bbox[1]) - float(bbox[3])/2.0
        right  = float(bbox[0]) + float(bbox[2])/2.0
        bottom = float(bbox[1]) + float(bbox[3])/2.0

        id_name = comm.IMVT_CLS_NAMES[int(id)]

        info = ""
        if len(items) == 5:
            info = f"{id_name} {left} {top} {right} {bottom}"
        elif len(items) == 6:
            info = f"{id_name} {prob} {left} {top} {right} {bottom}"

        fmt_line.append(info)

    comf.ensure_file_dir(target)
    comf.write_list(target, fmt_line)

if __name__ == '__main__':
    yolo_labels = comf.get_files_in_current_dir("/home/leo/coco/labels_yolo/val2017")
    true_labels = comf.get_files_in_current_dir("/home/leo/coco/labels/val2017")

    empty_list = []
    for each in yolo_labels:
        lines = comf.read_list(each)
        if len(lines) <= 0:
            empty_list.append(each)
            #print(each, "NG no line")
            continue

        for idx, line in enumerate(lines):
            items = line.split()
            if len(items) != 6:
                print('idx:', idx, 'fields:', len(items), line)
                print(each, "NG")
                exit()

    truth_list = []
    for each in true_labels:
        lines = comf.read_list(each)
        if len(lines) <= 0:
            print(each, "NG no line")
            continue

        truth_list.append(each)
        for idx, line in enumerate(lines):
            items = line.split()
            if len(items) != 5:
                print('idx:', idx, 'fields:', len(items), line)
                print(each, "NG")
                exit()

    for gt_file in truth_list:
        # "/home/leo/coco/labels/val2017/xxxxx.txt
        # "/home/leo/coco/val_gt_labels/val2017/xxxxx.txt
        gt_new =  gt_file.replace("labels", "val_gt_labels")

        # "/home/leo/coco/labels/val2017/xxxxx.txt
        # "/home/leo/coco/labels/labels_yolo/xxxxx.txt
        yolo =  gt_file.replace("labels", "labels_yolo")
        if yolo in empty_list:
            continue

        comf.ensure_file_dir(gt_new)
        copy_and_convert_cxcywh_xyxy_file(gt_file, gt_new)

        yolo_old = gt_file.replace("labels", "labels_yolo")
        yolo_new = gt_file.replace("labels", "val_yolo_labels")

        comf.ensure_file_dir(yolo_new)
        copy_and_convert_cxcywh_xyxy_file(yolo_old, yolo_new)

        print(gt_new, yolo_new)

    print("done")