import cv2
import os, sys

import com_detection as comm
import com_files as comf

def copy_hagrid_label_to_new_path(filenames:list):
    #   /home/leo/hand_fullset/train_labels/call/5ade9bf2-07bd-4d9b-b3a3-fc736267cfeb_mp_hand.txt  =>
    #   /home/leo/hand_sample/train/call/5ade9bf2-07bd-4d9b-b3a3-fc736267cfeb.txt
    for src_file in filenames:
        tar_file = src_file.replace("_mp_hand.txt", ".txt")
        tar_file = tar_file.replace('hand_fullset', 'hand_sample').replace('train_labels', 'train')
        comf.ensure_file_dir(tar_file)
        comf.copy_one_file(src_file, tar_file)
    print(f"select_labels_files:{len(filenames)}")

def rescale_hagrid_image_to_new_path(filenames:list):
    for src_filepath in filenames:
        #   /home/leo/hand_fullset/train_labels/call/5ade9bf2-07bd-4d9b-b3a3-fc736267cfeb_mp_hand.txt   =>
        #   /home/leo/hand_fullset/train/call/5ade9bf2-07bd-4d9b-b3a3-fc736267cfeb.jpg                  =>
        #   /home/leo/hand_sample/train/call/5ade9bf2-07bd-4d9b-b3a3-fc736267cfeb.jpg
        src_file = src_filepath.replace("_mp_hand.txt", ".jpg").replace('train_labels', 'train')
        tar_file = src_file.replace('hand_fullset', 'hand_sample')
        comf.ensure_file_dir(tar_file)
        frame = cv2.imread(src_file)
        frame = cv2.resize(frame, comm.YOLO_IMAGE_SIZE)
        cv2.imwrite(tar_file, frame)
    print(f"scale_image_files:{len(filenames)}")

if __name__ == '__main__':
    label_path  = "/home/leo/hand_fullset/train_labels"

    # 1, copy label
    #   /home/leo/hand_fullset/train_labels/call/5ade9bf2-07bd-4d9b-b3a3-fc736267cfeb_mp_hand.txt  =>
    #   /home/leo/hand_sample/train/call/5ade9bf2-07bd-4d9b-b3a3-fc736267cfeb.txt

    # 2, copy and rescale image(416x416)
    #   /home/leo/hand_fullset/train/call/5ade9bf2-07bd-4d9b-b3a3-fc736267cfeb.jpg  =>
    #   /home/leo/hand_sample/train/call/5ade9bf2-07bd-4d9b-b3a3-fc736267cfeb.jpg           //(416x416)

    all_files = comf.get_all_files_in_dir([label_path])
    mp_txt_files = comf.select_file_with_pattern(all_files, ['mp_hand'])

    copy_hagrid_label_to_new_path(mp_txt_files)
    rescale_hagrid_image_to_new_path(mp_txt_files)

