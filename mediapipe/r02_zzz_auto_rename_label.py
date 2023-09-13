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


def remove_all_file_with_pattern(path, filenames:list, pattern = "mp_hand"):
    remove_files = []
    for each in filenames:
        if "mp_hand" in each:
            remove_files.append(each)

    if len(remove_files) > 0:
        for onefile in remove_files:
            filename_fullpath = os.path.join(path, onefile)
            os.remove(filename_fullpath)
            print(filename_fullpath, "[removed]")

def rename_mp_hand_files_to_label_files(path, filenames:list):
    select_files = []
    remove_files = []
    for each in filenames:
        if "mp_hand" in each:
            select_files.append(each)
        else:
            remove_files.append(each)

    if len(select_files) > 0:
        for onefile in remove_files:
            filename_fullpath = os.path.join(path, onefile)
            os.remove(filename_fullpath)
    else:
        print("no *mp_hand.txt or all is processed")

    for onefile in select_files:
        filename = os.path.join(path, onefile)
        filename_fullpath = filename.replace("_mp_hand.txt", ".txt")
        os.rename(filename, filename_fullpath)

    print(f"select_files:{len(select_files)}")
    print(f"remove_files:{len(remove_files)}")

if __name__ == '__main__':
    #F:\hagrid\download\subsample\subsample\train_labels\call\5ade9bf2-07bd-4d9b-b3a3-fc736267cfeb_mp_hand.txt
    # ->
    #F:\hagrid\download\subsample\subsample\train_labels\call\5ade9bf2-07bd-4d9b-b3a3-fc736267cfeb.txt

    #path = "/home/leo/myhome/hagrid/download/subsample/train_labels"
    path = "/home/leo/hand_fullset/train_labels"

    all_files = get_all_files_in_dir([path])

    remove_all_file_with_pattern(path, all_files, pattern="mp_hand")

    #rename_mp_hand_files_to_label_files(path, all_files)
