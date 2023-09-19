import cv2
import mediapipe as mp
import os, time

import com_files as comf

if __name__ == '__main__':
    path  = "/home/leo/coco/labels"
    all_files = comf.get_all_files_in_dir([path])
    label_files = comf.select_file_with_pattern(all_files, ['txt'])

    # ensure image and label in the same dir:
    image_files_with_label = []
    for labelf in label_files:
        #/home/leo/coco/images/train2017/000000300024.jpg
        #/home/leo/coco/labels/train2017/000000300024.txt
        imagef = labelf.replace("labels", "images").replace(".txt", ".jpg")
        if os.path.isfile(imagef):
            image_files_with_label.append(imagef)

    output_name = "sel_coco_filelist.txt"
    comf.write_list_to_file(image_files_with_label,  os.path.join("./output",  output_name))
