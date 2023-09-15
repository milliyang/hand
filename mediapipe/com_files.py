import cv2
import os, time, sys
import shutil

def get_files_in_current_dir(dir):
    if not os.path.isdir(dir):
        return None
    files = os.listdir(dir)
    select_files = []
    for onefile in files:
        select_files.append(os.path.join(dir, onefile))
    return select_files

def get_all_files_in_dir(dir_list =[]):
    fullset_files = []
    for one_dir in dir_list:
        for entry in os.listdir(one_dir):
            path = os.path.join(one_dir, entry)
            if os.path.isdir(path):
                imgs = get_files_in_current_dir(path)
                fullset_files.extend(imgs)
                print(len(imgs), "totals:", len(fullset_files), path)
            if os.path.isfile(path):
                fullset_files.append(path)
    return fullset_files

def select_file_with_pattern(files:list, pattern = ["mp_hand"]):
    select_files = []
    for each in files:
        for pp in pattern:
            if pp in each:
                select_files.append(each)

    file_set = set(select_files)
    return list(file_set)

def remove_all_files(files : list):
    for afile in files:
        if os.path.isfile(afile):
            os.remove(afile)

def ensure_file_dir(filepath):
    try:
        pathname = os.path.dirname(filepath)
        os.makedirs(pathname)
    except:
        pass

def copy_one_file(source, target):
    if os.path.isfile(target):
        os.remove(target)

    try:
        shutil.copy(source, target)
    except IOError as e:
        print("Unable to copy file.", e)
    except:
        print("Unexpected error:", sys.exc_info())

def read_list(filename):
    #filename = "/home/leo/myhome/dataset/selected_voc_train.txt"
    afile = open(filename)
    images = afile.readlines()
    images = [ima.strip() for ima in images]
    return images

def write_list(filename, alist: list):
    file = open(filename, "w")
    for item in alist:
        file.write(item)
        file.write("\n")
    file.close()

def write_list_to_file(alist: list, filename="filelist.txt"):
    write_list(filename, alist)
    print("generate:", filename, "  num:", len(alist))

def read_filelist(filename):
    return read_list(filename)
