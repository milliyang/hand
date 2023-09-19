import os
import random
import com_detection as comm
import com_files as comf
import sys

input_file = "./output/fullset_filelist.txt"
img_files = comf.read_filelist(input_file)

random.shuffle(img_files)
comf.write_list(input_file, img_files)
