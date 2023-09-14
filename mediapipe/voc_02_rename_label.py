import cv2
import os, sys
import shutil

import com_detection as comm
import com_files as comf

def copy_voc_label_to_new_dir(select_files:list):
    #   /mnt/214cd1a6-9d1b-4b2a-9770-425b64e6884f/myhome/dataset/VOCdevkit/VOC2012/labels_voc/2011_001449_mp_hand.txt  =>
    #   /mnt/214cd1a6-9d1b-4b2a-9770-425b64e6884f/myhome/dataset/VOCdevkit/VOC2012/labels/2011_001449.txt
    for onefile in select_files:
        new_file = onefile.replace("_mp_hand.txt", ".txt").replace("labels_voc", "labels")
        comf.ensure_file_dir(new_file)
        comf.copy_one_file(onefile, new_file)

def generate_voc_filelist(select_files, filename="filelist.txt"):
    #   /mnt/214cd1a6-9d1b-4b2a-9770-425b64e6884f/myhome/dataset/VOCdevkit/VOC2012/labels/2011_001449_mp_hand.txt  =>
    #   /mnt/214cd1a6-9d1b-4b2a-9770-425b64e6884f/myhome/dataset/VOCdevkit/VOC2012/JPEGImages/2011_001449.jpg
    file = open(filename, "w")
    count = 0
    for labelf in select_files:
        imagef = labelf.replace("_mp_hand.txt", ".jpg").replace("labels_voc", "JPEGImages")
        file.write(imagef)
        file.write("\n")
        count += 1
    file.close()
    print("generate:", filename)
    print("files:", count)

if __name__ == '__main__':
    path_list = [
        '/mnt/214cd1a6-9d1b-4b2a-9770-425b64e6884f/myhome/dataset/VOCdevkit/VOC2012/',
        '/mnt/214cd1a6-9d1b-4b2a-9770-425b64e6884f/myhome/dataset/VOCdevkit/VOC2007/'
    ]
    output_path = "/mnt/214cd1a6-9d1b-4b2a-9770-425b64e6884f/myhome/dataset/"
    output_name = "sel_voc_filelist.txt"

    # 1, copy label
    #   /mnt/214cd1a6-9d1b-4b2a-9770-425b64e6884f/myhome/dataset/VOCdevkit/VOC2012/labels_voc/2011_001449_mp_hand.txt  =>
    #   /mnt/214cd1a6-9d1b-4b2a-9770-425b64e6884f/myhome/dataset/VOCdevkit/VOC2012/labels/2011_001449.txt
    all_files = comf.get_all_files_in_dir(path_list)

    select_files = comf.select_file_with_pattern(all_files, pattern=["mp_hand"])
    print(f"mp_hand.txt  files:{len(select_files)}")

    copy_voc_label_to_new_dir(select_files)

    generate_voc_filelist(select_files, filename=os.path.join(output_path, output_name))
    generate_voc_filelist(select_files, filename=os.path.join("./output",  output_name))
    
