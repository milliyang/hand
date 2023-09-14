import cv2
import mediapipe as mp
import os, time

import com_files as comf

if __name__ == '__main__':
    path  = "/home/leo/hand_sample/train"
    all_files = comf.get_all_files_in_dir([path])
    img_files = comf.select_file_with_pattern(all_files, ['jpg', 'png'])
    label_files = comf.select_file_with_pattern(all_files, ['txt'])

    # ensure image and label in the same dir:
    image_files_with_label = []
    for imagef in img_files:
        imagef_base = os.path.basename(imagef)
        imagef_dir  = os.path.dirname(imagef)
        labelf_base = imagef_base.replace(".jpg", ".txt").replace(".png", ".txt")
        labelf = os.path.join(imagef_dir, labelf_base)
        if labelf in label_files:
            image_files_with_label.append(imagef)

    output_path = "/mnt/214cd1a6-9d1b-4b2a-9770-425b64e6884f/myhome/dataset/"
    output_name = "sel_hagrid_filelist.txt"
    comf.write_list_to_file(image_files_with_label,  os.path.join("./output",  output_name))
    comf.write_list_to_file(image_files_with_label,  os.path.join(output_path, output_name))
