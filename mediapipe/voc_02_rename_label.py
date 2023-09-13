import cv2
import os, sys
import shutil

YOLO_IMAGE_SIZE = (416,416)

def get_files_in_current_dir(dir):
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
                imgs = get_files_in_current_dir(path)
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

def select_file_with_pattern(files:list, pattern = ["mp_hand"]):
    select_files = []
    for each in files:
        for pp in pattern:
            if pp in each:
                select_files.append(each)
    return select_files

def copy_voc_label_to_new_dir(select_files:list):
    #   /mnt/214cd1a6-9d1b-4b2a-9770-425b64e6884f/myhome/dataset/VOCdevkit/VOC2012/labels_voc/2011_001449_mp_hand.txt  =>
    #   /mnt/214cd1a6-9d1b-4b2a-9770-425b64e6884f/myhome/dataset/VOCdevkit/VOC2012/labels/2011_001449.txt
    for onefile in select_files:
        new_file = onefile.replace("_mp_hand.txt", ".txt").replace("labels_voc", "labels")
        ensure_file_dir(new_file)
        copy_one_file(onefile, new_file)

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
    all_files = get_all_files_in_dir(path_list)

    select_files = select_file_with_pattern(all_files, pattern=["mp_hand"])
    print(f"mp_hand.txt  files:{len(select_files)}")

    copy_voc_label_to_new_dir(select_files)

    generate_voc_filelist(select_files, filename=os.path.join(output_path, output_name))
    generate_voc_filelist(select_files, filename=output_name)
