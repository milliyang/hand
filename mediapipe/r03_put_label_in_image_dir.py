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

def copy_file(src, target):
    cmd = f"cp {src} {target}"
    #print(cmd)
    os.system(cmd)

if __name__ == '__main__':
    #label_path = "/home/leo/myhome/hagrid/download/subsample/train_labels"
    #image_path = "/home/leo/myhome/hagrid/download/subsample/train"

    label_path = "/home/leo/hand_fullset/train_labels"
    #image_path = "/home/leo/hand_fullset/train"
    all_files = get_all_files_in_dir([label_path])

    #/home/leo/myhome/hagrid/download/subsample/train_labels/call/577c276d-f78a-45a8-b31d-2d4e59d31a89.txt
    # -->
    #/home/leo/myhome/hagrid/download/subsample/train/call/577c276d-f78a-45a8-b31d-2d4e59d31a89.txt
    #/home/leo/myhome/hagrid/download/subsample/train/call/577c276d-f78a-45a8-b31d-2d4e59d31a89.jpg

    for each in all_files:
        target = each.replace("train_labels", "train")
        copy_file(each, target)

    #generate jpg list
    with open("hand_fullset.txt", "w+") as file:
        for each in all_files:
            target = each.replace("train_labels", "train").replace(".txt", ".jpg")
            file.write(f"{target}\n")