import cv2
import os, sys
import shutil
import com_detection as comm
import com_files as comf

def copy_wider_label_to_image_path(filenames:list):
    for src_file in filenames:
        tar_file = src_file.replace("_mp_hand.txt", ".txt")
        tar_file = tar_file.replace('labels_face', 'labels')
        comf.ensure_file_dir(tar_file)
        comf.copy_one_file(src_file, tar_file)
    print(f"copy_labels_files:{len(filenames)}")

def generate_filelist(image_files, filename="filelist.txt"):
    #   /mnt/214cd1a6-9d1b-4b2a-9770-425b64e6884f/myhome/dataset/VOCdevkit/VOC2012/labels/2011_001449_mp_hand.txt  =>
    #   /mnt/214cd1a6-9d1b-4b2a-9770-425b64e6884f/myhome/dataset/VOCdevkit/VOC2012/JPEGImages/2011_001449.jpg
    file = open(filename, "w")
    count = 0
    for imagef in image_files:
        file.write(imagef)
        file.write("\n")
        count += 1
    file.close()
    print("generate:", filename)
    print("files:", count)

if __name__ == '__main__':
    image_path  = "/home/leo/myhome/WIDER_train/images"
    label_path  = "/home/leo/myhome/WIDER_train/labels_face"
    output_path = "/home/leo/myhome/WIDER_train/labels"

    # 1. remove all txt file from "/home/leo/myhome/WIDER_train/images"
    #
    # 2. darknet:
    #    copy mp_hand.txt -> to labels dir:
    #    /home/leo/myhome/WIDER_train/labels_face/0--Parade/0_Parade_Parade_0_9_mp_hand.txt =>
    #    /home/leo/myhome/WIDER_train/labels/0--Parade/0_Parade_Parade_0_9.txt

    all_files = comf.get_all_files_in_dir([image_path])
    txt_files = comf.select_file_with_pattern(all_files, ['.txt'])
    comf.remove_all_files(txt_files)

    print("\n")
    image_files = comf.get_all_files_in_dir([image_path])

    labels_files = comf.get_all_files_in_dir([label_path])
    labels_files = comf.select_file_with_pattern(labels_files, ['mp_hand'])

    print("\n")
    copy_wider_label_to_image_path(labels_files)

    # ensure image and label in the same dir:
    vol_image_list = "/home/leo/myhome/WIDER_train/wider_filelists.txt"
    images = comm.read_filelist(vol_image_list)

    image_files = comf.get_all_files_in_dir([image_path])
    label_files = comf.get_all_files_in_dir([output_path])

    image_files_with_label = []
    for imagef in images:
        #/home/leo/myhome/WIDER_train/images/0--Parade/0_Parade_marchingband_1_849.jpg
        # make sure: 0_Parade_marchingband_1_849.jpg, 0_Parade_marchingband_1_849.txt both exist
        labelf = imagef.replace(".jpg", ".txt").replace("images", "labels")
        if (labelf in label_files) and (imagef in image_files):
            image_files_with_label.append(imagef)

    output_path = "/mnt/214cd1a6-9d1b-4b2a-9770-425b64e6884f/myhome/dataset/"
    output_name = "sel_wider_filelist.txt"
    generate_filelist(image_files_with_label,  os.path.join("./output",  output_name))
    generate_filelist(image_files_with_label,  os.path.join(output_path, output_name))
