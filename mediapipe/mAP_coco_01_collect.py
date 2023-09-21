import cv2
import mediapipe as mp
import os, time

import com_files as comf

def collect_yolo_detect_result(infile):
    imagefiles = comf.read_filelist(infile)

    for imgf in imagefiles:
        #/home/leo/coco/images/train2017/000000300024.jpg
        #/home/leo/coco/images/train2017/000000300024.jpg.yolo.txt
        labelf = imgf + ".yolo.txt"
        if os.path.isfile(labelf) and os.path.isfile(imgf):

            #/home/leo/coco/label_yolo/train2017/000000300024.txt
            outfile = imgf.replace("images","labels_yolo").replace(".jpg",".txt")
            comf.ensure_file_dir(outfile)
            comf.copy_one_file(labelf, outfile)

if __name__ == '__main__':
    infile = "/home/leo/coco/coco_val2017_filelists.txt"
    collect_yolo_detect_result(infile)
