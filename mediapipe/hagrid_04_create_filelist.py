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

def select_file_with_pattern(files:list, pattern = ["mp_hand"]):
    select_files = []
    for each in files:
        for pp in pattern:
            if pp in each:
                select_files.append(each)
    return select_files

def generate_filelist(image_files, filename="filelist.txt"):
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
    path  = "/home/leo/hand_sample/train"
    all_files = get_all_files_in_dir([path])
    img_files = select_file_with_pattern(all_files, ['jpg', 'png'])
    label_files = select_file_with_pattern(all_files, ['txt'])

    # ensure image and label in the same dir:
    image_files_with_label = []
    for imagef in img_files:
        imagef_base = os.path.basename(imagef)
        imagef_dir  = os.path.dirname(imagef)
        labelf_base = imagef_base.replace(".jpg", ".txt").replace(".png", ".txt")
        labelf = os.path.join(imagef_dir, labelf_base)
        if labelf in label_files:
            image_files_with_label.append(imagef)

    output_path = "/mnt/214cd1a6-9d1b-4b2a-9770-425b64e6884f/myhome/dataset/"
    output_name = "sel_hagrid_filelist.txt"
    generate_filelist(image_files_with_label,  os.path.join("./output",  output_name))
    generate_filelist(image_files_with_label,  os.path.join(output_path, output_name))
