import cv2
import mediapipe as mp
import os, time
import com_detection as comm
import com_files as comf
import sys

if __name__ == '__main__':
    DATA_TYPE = "train2017"
    #DATA_TYPE = "val2017"
    fullset_file = f"/home/leo/coco/coco_train2017_filelists.txt"
    used_file = f"./output/sel_train2017_coco_filelist.txt"

    fullsets = comf.read_filelist(fullset_file)
    used = comf.read_filelist(used_file)

    max_idx = 0
    max_filename = ''
    for idx, each in enumerate(fullsets):
        if each in used:
            if idx > max_idx:
                max_idx = idx
                max_filename = each

    print(f'max_idx:{max_idx}')
    print(f'max_filename:{max_filename}')
