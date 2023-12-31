import cv2
import mediapipe as mp
import os, time

import com_files as comf

if __name__ == '__main__':
    label_files = comf.get_files_in_current_dir(f"/home/leo/custom_face/labels")

    # ensure image and label in the same dir:
    image_files_with_label = []
    for labelf in label_files:
        #/home/leo/coco/images/train2017/000000300024.jpg
        #/home/leo/coco/labels/train2017/000000300024.txt
        imagef = labelf.replace("labels", "images").replace(".txt", ".JPG")
        if os.path.isfile(imagef):
            image_files_with_label.append(imagef)
    output_name = f"sel_train_cus_filelist.txt"
    comf.write_list_to_file(image_files_with_label,  os.path.join("./output",  output_name))
