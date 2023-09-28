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
    #remove old data
    files = comf.remove_all_files_in_dir(["/home/leo/coco/val_gt_labels", "/home/leo/coco/val_yolo_labels"])

    yolo_labels = []
    yolo_labels0 = comf.get_files_in_current_dir("/home/leo/coco/labels_yolo/val2017")
    yolo_labels1 = comf.get_files_in_current_dir("/home/leo/coco/labels_yolo/train2017")

    yolo_labels.extend(yolo_labels0)
    yolo_labels.extend(yolo_labels1)
    print("yolo_labels:", len(yolo_labels))

    true_labels = []
    for lab in yolo_labels:
        #/home/leo/coco/labels_yolo/val2017/xxxxx.txt
        #/home/leo/coco/labels_yolo/train2017/xxxxx.txt
        true_lab = lab.replace("labels_yolo", "labels")
        if os.path.isfile(true_lab):
            true_labels.append(true_lab)

    print("true_labels:", len(true_labels))

    empty_list = []
    for each in yolo_labels:
        if not os.path.isfile(each):
            continue
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
        if not os.path.isfile(each):
            continue

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

    count = 0
    for gt_file in truth_list:
        # "/home/leo/coco/labels/val2017/xxxxx.txt
        # "/home/leo/coco/labels/labels_yolo/xxxxx.txt
        yolo =  gt_file.replace("labels", "labels_yolo")
        if yolo in empty_list:
            continue

        # "/home/leo/coco/labels/val2017/xxxxx.txt =>
        # "/home/leo/coco/val_gt_labels/val2017/xxxxx.txt
        gt_new =  gt_file.replace("labels", "val_gt_labels")

        yolo_old = gt_file.replace("labels", "labels_yolo")
        yolo_new = gt_file.replace("labels", "val_yolo_labels")

        #hack train to val dir
        if "train2017" in gt_new:
            gt_new =  gt_new.replace("train2017", "val2017")
            gt_new =  gt_new.replace(".txt", ".train2017.txt")

            yolo_new =  yolo_new.replace("train2017", "val2017")
            yolo_new =  yolo_new.replace(".txt", ".train2017.txt")

        comf.ensure_file_dir(gt_new)
        copy_and_convert_cxcywh_xyxy_file(gt_file, gt_new)

        comf.ensure_file_dir(yolo_new)
        copy_and_convert_cxcywh_xyxy_file(yolo_old, yolo_new)

        print(gt_new, yolo_new)
        count+=1

    print("done:", count)