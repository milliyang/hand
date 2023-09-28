import cv2
import mediapipe as mp
import os, time
import com_detection as comm
import com_files as comf
import sys

if __name__ == '__main__':
    DATA_TYPE = "train2017"
    #DATA_TYPE = "val2017"
    include_file = f"./output/sel_{DATA_TYPE}_coco_filelist.txt"
    exclude_files = [
        f"./output/exclude_coco_filelist.txt",              #human manual select invalid data
        f"./output/sel_val2017_coco_filelist.txt",          #move val   data for testing
        f"./output/sel_val2017_coco_filelist_train.txt",    #move train data for testing
        #
        f'./output/sel_val2017_coco_filelist_all.txt',      #duplicated, but it's OK
    ]

    output_file = "./output/fullset_filelist.txt"           #for training

    include = comf.read_filelist(include_file)
    exclude = []

    for each in exclude_files:
        alist = comf.read_filelist(each)
        exclude.extend(alist)

    print(f"include:{len(include)}")
    print(f"exclude:{len(exclude)}")
    filelists = []
    for each in include:
        if each in exclude:
            continue
        filelists.append(each)

    comf.write_list_to_file(filelists)