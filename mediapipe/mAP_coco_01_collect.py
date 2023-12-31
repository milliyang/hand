import cv2
import mediapipe as mp
import os, time
import argparse
import com_files as comf

INPUT_DEF = "/home/leo/coco/coco_val2017_filelists.txt"

parser = argparse.ArgumentParser()
parser.add_argument("-filelist", type=str, help="input filelists.txt", default=INPUT_DEF)
args = parser.parse_args()

def collect_yolo_detect_result(infile):
    print("infile:", infile)

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
    #
    #infile = "/home/leo/coco/coco_val2017_filelists.txt"

    #train 数据极大, 部分未用于train的数据, 用于验证
    #infile = "/home/leo/myhome/hand/mediapipe/output/sel_val2017_coco_filelist_all.txt"

    infile = args.filelist
    collect_yolo_detect_result(infile)
