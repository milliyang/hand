from pycocotools.coco import COCO
import numpy as np

import cv2
import mediapipe as mp
import os, time
import com_detection as comm
import com_files as comf

#pip install pycocotools

COCO_PATH = "/home/leo/coco"
COCO_TYPE='train2017'
#COCO_TYPE='val2017'
annFile='{}/annotations/instances_{}.json'.format(COCO_PATH, COCO_TYPE)

LABELS_PREFIX = "labels_coco"

IGNORE_FACE_OVERLAP = True

# initialize COCO api for instance annotations
coco=COCO(annFile)

def coco_show_categories(coco):
    # display COCO categories and supercategories
    cats = coco.loadCats(coco.getCatIds())
    nms=[cat['name'] for cat in cats]
    print('COCO categories: \n{}\n'.format(' '.join(nms)))
    nms = set([cat['supercategory'] for cat in cats])
    print('COCO supercategories: \n{}'.format(' '.join(nms)))

def coco_get_image_ids_with_cat(cats = ['person']):
    # get all images containing given categories, select one at random
    #catIds = coco.getCatIds(catNms=['person','dog','skateboard'])
    #catIds = coco.getCatIds(catNms=['person'])
    catIds = coco.getCatIds(catNms=cats)
    imgIds = coco.getImgIds(catIds=catIds)
    print('len:', len(imgIds))
    return catIds, imgIds

def calc_iou(box1, box2):
    """
    :param box1: = [xmin1, ymin1, xmax1, ymax1]
    :param box2: = [xmin2, ymin2, xmax2, ymax2]
    :return: 
    """
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    #
    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)  # b1 area
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)  # b2 area
 
    #
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)
 
    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)
    a1 = w * h  # C∩G
    a2 = s1 + s2 - a1
    iou = a1 / a2 #iou = a1/ (s1 + s2 - a1)
    return iou

def coco_get_face_info_from_yoloface_fast_predictions(imagef, width, height):
    #show coco yoloface_fast_predictions

    #"/home/leo/coco/images/val2017/000000036936.jpg"
    #"/home/leo/coco/yoloface_fast_predictions/val2017/xxxx.csv"
    cocoface_label = imagef.replace("images", "yoloface_fast_predictions") + ".csv"
    facesline = comm.read_filelist(cocoface_label)

    facebox = []
    for face in facesline:
        items = face.split(",")
        if (len(items) != 6):
            continue
        #print(items[1:])
        #'/data/coco/val2017/000000036936.jpg,0.9891075,104.2295,414.4043,148.12671,440.77728'
        ppath, prob, y0, x0, y1, x1 = items
        x0 = float(x0) / width
        y0 = float(y0) / height
        x1 = float(x1) / width
        y1 = float(y1) / height
        prob = float(prob)
        w = x1 - x0
        h = y1 - y0
        #if w < comm.YOLO_FACE_MIN_SIZE or h < comm.YOLO_FACE_MIN_SIZE: continue
        if w < comm.YOLO_FACE_MIN_SIZE or h < comm.YOLO_FACE_MIN_SIZE:
            continue
        facebox.append([prob,x0,y0,x1,y1])

    overlap = False

    if len(facebox) > 1:
        idx_used = set()
        for ii in range(0, len(facebox)-1):
            box0 = facebox[ii][1:]
            for jj in range(ii+1, len(facebox)):
                box1 = facebox[jj][1:]
                iou = calc_iou(box0, box1)
                if iou > 0:
                    overlap = True
                    #print(ii, jj, box0, box1, 'overlap', 'iou:', iou)

    infos = []
    for face in facebox:
        prob,x0,y0,x1,y1 = face
        w = x1-x0
        h = y1-y0
        a_box = [x0, y0, w, h]
        info = [comm.YOLO_FACE_ID, comm.id_to_names(comm.YOLO_FACE_ID), prob, a_box]
        #print(info)
        infos.append(info)

    return overlap, infos

def coco_cv_show(catIds, imgIds):
    txtfile_cc = (0, 255, 120)
    coco_images = []
    for img_id in imgIds:
        img = coco.loadImgs(img_id)[0]

        imagef = f"{COCO_PATH}/images/{COCO_TYPE}/{img['file_name']}"
        #print('imagef', imagef)             # /home/leo/coco/images/val2017/000000413689.jpg
        #print('img_id', img_id)             # 413689
        #print('coco_url', img['coco_url'])  # http://images.cocodataset.org/val2017/000000108503.jpg

        frame = cv2.imread(imagef)
        height_ori, width_ori, _  = frame.shape

        frame = cv2.resize(frame, comm.YOLO_IMAGE_SIZE)
        height, width, _  = frame.shape

        # load and display instance annotations
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)

        object_labels = []

        for idx, ann in enumerate(anns):
            #print(idx)
            #print(' iscrowd', ann['iscrowd'])
            #print(ann)
            if ann['iscrowd']:
                continue

            coco_id = ann['category_id']
            coco_name = comm.COCO_ID_NAME_MAP[coco_id]
            a_box = ann['bbox']
            a_box = [a_box[0]/width_ori, a_box[1]/height_ori, a_box[2]/width_ori, a_box[3]/height_ori]

            if a_box[2] < comm.YOLO_OBJECT_MIN_SIZE or a_box[3] < comm.YOLO_OBJECT_MIN_SIZE:
                continue

            info  = [coco_id, coco_name, 1.0, a_box]
            comm.draw_info_on_image(frame, width, height, info, txtfile_cc, 1)

            #coco id to imvt_yolo_id
            if coco_name in comm.IMVT_CLS_NAMES:
                yolo_id = comm.IMVT_CLS_NAMES.index(coco_name)
                info  = [yolo_id, coco_name, 1.0, a_box]
                object_labels.append(comm.info_to_yolo_string(info))

        person_num = len(object_labels)

        face_overlap, face_infos = coco_get_face_info_from_yoloface_fast_predictions(imagef, width_ori, height_ori)

        if IGNORE_FACE_OVERLAP and face_overlap:
            continue    #SKIP OVERLAP

        if len(face_infos) > 0:
            for info in face_infos:
                object_labels.append(comm.info_to_yolo_string(info))

        if len(object_labels) > 0:
            labelf = imagef.replace("images", LABELS_PREFIX).replace(".jpg", ".txt")
            comf.ensure_file_dir(labelf)
            comf.write_list(labelf, object_labels)
            print('labelf', labelf, 'faces:', len(face_infos), 'person:', person_num)
            coco_images.append(imagef)

        #cv2.imshow('Coco', frame)
        #if (cv2.waitKey(1000*3) & 0xFF == ord(comm.EXIT_KEY)): break

    coco_image_list = os.path.join(COCO_PATH, f"coco_{COCO_TYPE}_filelists.txt")
    comf.write_list_to_file(coco_images, coco_image_list)

coco_show_categories(coco)

catIds, imgIds = coco_get_image_ids_with_cat(cats = ['person'])

#coco_show_image(catIds, imgIds)
coco_cv_show(catIds, imgIds)