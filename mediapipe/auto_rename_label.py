import cv2
import mediapipe as mp
import os, time

def get_images_in_current_dir(dir):
    if not os.path.isdir(dir):
        return None
    files = os.listdir(dir)
    select_files = []
    for onefile in files:
        select_files.append(os.path.join(dir, onefile))
    return select_files

def get_all_files_in_dir(dir_list =[]):
    images = []
    for one_dir in dir_list:
        for entry in os.listdir(one_dir):
            path = os.path.join(one_dir, entry)
            if os.path.isdir(path):
                imgs = get_images_in_current_dir(path)
                images.extend(imgs)
                print(len(imgs), "totals:", len(images), path)
    return images

if __name__ == '__main__':
    #F:\hagrid\download\subsample\subsample\train_labels\call\5ade9bf2-07bd-4d9b-b3a3-fc736267cfeb_auto_hand.txt
    # ->
    #F:\hagrid\download\subsample\subsample\train_labels\call\5ade9bf2-07bd-4d9b-b3a3-fc736267cfeb.txt

    path = "F:\\hagrid\\download\\subsample\\subsample\\train_labels"

    all_files = get_all_files_in_dir([path])

    select_files = []
    remove_files = []
    for each in all_files:
        if "auto_hand" in each:
            select_files.append(each)
        else:
            remove_files.append(each)

    for onefile in remove_files:
        filename_new = os.path.join(path, onefile)
        os.remove(filename_new)

    for onefile in select_files:
        filename = os.path.join(path, onefile)
        filename_new = filename.replace("_auto_hand.txt", ".txt")
        os.rename(filename, filename_new)
