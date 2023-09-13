import cv2
import os, sys
import shutil

YOLO_IMAGE_SIZE = (416,416)

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

def copy_one_file(source, target):
    if os.path.isfile(target):
        os.remove(target)

    try:
        shutil.copy(source, target)
    except IOError as e:
        print("Unable to copy file.", e)
    except:
        print("Unexpected error:", sys.exc_info())

def ensure_file_dir(filepath):
    try:
        pathname = os.path.dirname(filepath)
        os.makedirs(pathname)
    except:
        pass

def copy_hagrid_label_to_new_path(filenames:list, pattern = "mp_hand"):
    #   /home/leo/hand_fullset/train_labels/call/5ade9bf2-07bd-4d9b-b3a3-fc736267cfeb_mp_hand.txt  =>
    #   /home/leo/hand_sample/train/call/5ade9bf2-07bd-4d9b-b3a3-fc736267cfeb.txt
    select_files = []
    for each in filenames:
        if pattern in each:
            select_files.append(each)

    for src_file in select_files:
        tar_file = src_file.replace("_mp_hand.txt", ".txt")
        tar_file = tar_file.replace('hand_fullset', 'hand_sample').replace('train_labels', 'train')
        ensure_file_dir(tar_file)

        copy_one_file(src_file, tar_file)
        #print(src_file, '->')
        #print(tar_file)
        #break
    print(f"select_labels_files:{len(select_files)}")


def rescale_hagrid_image_to_new_path(filenames:list, pattern = "mp_hand"):
    #   /home/leo/hand_fullset/train_labels/call/5ade9bf2-07bd-4d9b-b3a3-fc736267cfeb_mp_hand.txt  =>
    #   /home/leo/hand_sample/train/call/5ade9bf2-07bd-4d9b-b3a3-fc736267cfeb.txt
    select_files = []
    for each in filenames:
        if pattern in each:
            select_files.append(each)

    for src_filepath in select_files:
        #   /home/leo/hand_fullset/train_labels/call/5ade9bf2-07bd-4d9b-b3a3-fc736267cfeb_mp_hand.txt   =>
        #   /home/leo/hand_fullset/train/call/5ade9bf2-07bd-4d9b-b3a3-fc736267cfeb.jpg                  =>
        #   /home/leo/hand_sample/train/call/5ade9bf2-07bd-4d9b-b3a3-fc736267cfeb.jpg
        src_file = src_filepath.replace("_mp_hand.txt", ".jpg").replace('train_labels', 'train')
        tar_file = src_file.replace('hand_fullset', 'hand_sample')
        ensure_file_dir(tar_file)
        frame = cv2.imread(src_file)
        frame = cv2.resize(frame, YOLO_IMAGE_SIZE)
        cv2.imwrite(tar_file, frame)
        #print(src_file, '->')
        #print(tar_file)
        #break

    print(f"scale_image_files:{len(select_files)}")

if __name__ == '__main__':
    label_path  = "/home/leo/hand_fullset/train_labels"

    # 1, copy label
    #   /home/leo/hand_fullset/train_labels/call/5ade9bf2-07bd-4d9b-b3a3-fc736267cfeb_mp_hand.txt  =>
    #   /home/leo/hand_sample/train/call/5ade9bf2-07bd-4d9b-b3a3-fc736267cfeb.txt

    # 2, copy and rescale image(416x416)
    #   /home/leo/hand_fullset/train/call/5ade9bf2-07bd-4d9b-b3a3-fc736267cfeb.jpg  =>
    #   /home/leo/hand_sample/train/call/5ade9bf2-07bd-4d9b-b3a3-fc736267cfeb.jpg           //(416x416)

    all_files = get_all_files_in_dir([label_path])

    copy_hagrid_label_to_new_path(all_files, pattern="mp_hand")
    rescale_hagrid_image_to_new_path(all_files, pattern="mp_hand")

