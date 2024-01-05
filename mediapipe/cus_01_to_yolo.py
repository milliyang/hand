import cv2
import numpy as np
import os, time
import com_detection as comm
import com_files as comf
#
from xml.dom.minidom import parse
import xml.dom.minidom

PATH_FULL_IMG = "/home/leo/custom_face/fullsize_images"
PATH_OUTPUT   = "/home/leo/custom_face"

files =  comf.get_files_in_current_dir(PATH_FULL_IMG)
xml_files = comf.select_file_with_pattern(files, [".xml"])
xml_files.sort()

def show_one_image(xml_file):
    print("xml_file", xml_file)
    DOMTree = xml.dom.minidom.parse(xml_file)
    collection = DOMTree.documentElement

    xml_w    = collection.getElementsByTagName("width")[0].childNodes[0].data
    xml_h    = collection.getElementsByTagName("height")[0].childNodes[0].data
    xml_path = collection.getElementsByTagName("path")[0].childNodes[0].data

    #print("xml_path",   xml_path)
    #print("xml_w",      xml_w)
    #print("xml_h",      xml_h)

    imagef = xml_file.replace(".xml", ".JPG")
    frame = cv2.imread(imagef)
    h_ori, w_ori, _  = frame.shape
    IMG_SIZE = 2*416
    frame = cv2.resize(frame, (IMG_SIZE,IMG_SIZE))
    height, width, _  = frame.shape

    frame_output = cv2.resize(frame, (IMG_SIZE,IMG_SIZE))

    object_labels = []

    objs = collection.getElementsByTagName("object")
    for obj in objs:
        xml_name = obj.getElementsByTagName("name")[0].childNodes[0].data
        xml_xmin = float(obj.getElementsByTagName("xmin")[0].childNodes[0].data)
        xml_ymin = float(obj.getElementsByTagName("ymin")[0].childNodes[0].data)
        xml_xmax = float(obj.getElementsByTagName("xmax")[0].childNodes[0].data)
        xml_ymax = float(obj.getElementsByTagName("ymax")[0].childNodes[0].data)
        #print(xml_name, xml_xmin, xml_ymin, xml_xmax, xml_ymax)

        obj_id = comm.YOLO_FACE_ID
        if xml_name == "face":
            obj_id = comm.YOLO_FACE_ID
        elif xml_name == "person":
            obj_id = comm.YOLO_HUMAN_ID
        coco_name = comm.id_to_names(obj_id)

        xywh = [xml_xmin/w_ori, xml_ymin/h_ori, (xml_xmax-xml_xmin) /w_ori, (xml_ymax-xml_ymin)/h_ori]
        info  = [obj_id, coco_name, 1.0, xywh]
        object_labels.append(comm.info_to_yolo_string(info))

        comm.draw_info_on_image(frame, width, height, info, comm.CC_TXTFILE, 1)

    if len(object_labels) > 0:
        #convert labels
        basename = os.path.basename(xml_file).replace(".xml", ".txt")
        labelf = os.path.join(PATH_OUTPUT, "labels", basename)
        comf.ensure_file_dir(labelf)
        comf.write_list(labelf, object_labels)
        #convert image
        basename = os.path.basename(xml_file).replace(".xml", ".JPG")
        output_img = os.path.join(PATH_OUTPUT, "images", basename)
        cv2.imwrite(output_img, frame_output)

    #cv2.imshow('Cus', frame)
    #if (cv2.waitKey(1000*1) & 0xFF == ord(comm.EXIT_KEY)): return True

    return False

for each in xml_files:
    quit = show_one_image(each)
    if quit:
        break
