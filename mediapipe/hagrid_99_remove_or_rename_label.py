import cv2
import os, time

import com_files as comf

def remove_all_file_with_pattern(path, filenames:list, pattern = "mp_hand"):
    remove_files = []
    for each in filenames:
        if pattern in each:
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
    path = "/home/leo/hand_fullset/train_labels"

    all_files = comf.get_all_files_in_dir([path])

    remove_all_file_with_pattern(path, all_files, pattern="mp_hand")
    #remove_all_file_with_pattern(path, all_files, pattern="txt")
    #rename_mp_hand_files_to_label_files(path, all_files)
