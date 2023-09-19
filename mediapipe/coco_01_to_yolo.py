from pycocotools.coco import COCO
import numpy as np

import cv2
import mediapipe as mp
import os, time
import com_detection as comm
import com_files as comf

#pip install pycocotools

COCO_PATH = "/home/leo/coco"
# ├── annotations
#     ├── captions_train2017.json
#     ├── captions_val2017.json
#     ├── instances_train2017.json
#     ├── instances_val2017.json
#     ├── person_keypoints_train2017.json
#     └── person_keypoints_val2017.json
# ├── train2017
# └── val2017

dataDir=COCO_PATH
#dataType='val2017'
dataType='train2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

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

def coco_cv_show(catIds, imgIds):

    txtfile_cc = (0, 255, 120)

    coco_images = []

    for img_id in imgIds:
        img = coco.loadImgs(img_id)[0]

        imagef = f"{dataDir}/images/{dataType}/{img['file_name']}"
        #print('imagef', imagef)             # /home/leo/coco/images/val2017/000000413689.jpg
        #print('img_id', img_id)             # 413689
        #print('coco_url', img['coco_url'])  # http://images.cocodataset.org/val2017/000000108503.jpg

        frame = cv2.imread(imagef)
        height_ori, width_ori, _  = frame.shape

        #frame = cv2.flip(frame, 1)

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

            info  = [coco_id, coco_name, 1.0, a_box]
            comm.draw_info_on_image(frame, width, height, info, txtfile_cc, 1)

            #coco id to imvt_yolo_id
            if coco_name in comm.IMVT_CLS_NAMES:
                yolo_id = comm.IMVT_CLS_NAMES.index(coco_name)
                info  = [yolo_id, coco_name, 1.0, a_box]
                object_labels.append(comm.info_to_yolo_string(info))

        if len(object_labels) > 0:
            labelf = imagef.replace("images", "labels_coco").replace(".jpg", ".txt")
            comf.ensure_file_dir(labelf)
            comf.write_list(labelf, object_labels)
            print('labelf', labelf)
            coco_images.append(imagef)

        #cv2.imshow('Coco', frame)
        #if (cv2.waitKey(1000*3) & 0xFF == ord(comm.EXIT_KEY)): break

    coco_image_list = os.path.join(COCO_PATH, "coco_filelists.txt")
    comf.write_list_to_file(coco_images, coco_image_list)

coco_show_categories(coco)

catIds, imgIds = coco_get_image_ids_with_cat(cats = ['person'])

#coco_show_image(catIds, imgIds)
coco_cv_show(catIds, imgIds)