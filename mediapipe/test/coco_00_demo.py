from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab

import cv2
import mediapipe as mp
import os, time
import com_detection as comm

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

pylab.rcParams['figure.figsize'] = (8.0, 10.0)


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
    imgIds = coco.getImgIds(catIds=catIds )
    #imgIds = coco.getImgIds(imgIds = [324158])
    print('len:', len(imgIds))
    #print(imgIds)
    return catIds, imgIds

def coco_show_image(catIds, imgIds):

    while True:
        img_id = imgIds[np.random.randint(0,len(imgIds))]
        img = coco.loadImgs(img_id)[0]

        # load and display image
        # I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
        # use url to load image
        I = io.imread(img['coco_url'])
        plt.axis('off')
        plt.imshow(I)
        #plt.show()

        # load and display instance annotations
        plt.imshow(I); plt.axis('off')
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        print('instance annotations', anns)
        coco.showAnns(anns)

        # initialize COCO api for person keypoints annotations
        #annFile = '{}/annotations/person_keypoints_{}.json'.format(dataDir,dataType)
        #coco_kps=COCO(annFile)

        ## load and display keypoints annotations
        #plt.imshow(I); plt.axis('off')
        #ax = plt.gca()
        #annIds = coco_kps.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        #anns = coco_kps.loadAnns(annIds)
        #print('keypoints annotations', anns)
        #coco_kps.showAnns(anns)

        # # initialize COCO api for caption annotations
        # annFile = '{}/annotations/captions_{}.json'.format(dataDir,dataType)
        # coco_caps=COCO(annFile)
        # # load and display caption annotations
        # annIds = coco_caps.getAnnIds(imgIds=img['id']);
        # anns = coco_caps.loadAnns(annIds)
        # print('caption annotations', anns)
        # coco_caps.showAnns(anns)
        plt.imshow(I); plt.axis('off'); plt.show()

def coco_cv_show(catIds, imgIds):

    txtfile_cc = (0, 255, 120)

    while True:
        img_id = imgIds[np.random.randint(0,len(imgIds))]
        img = coco.loadImgs(img_id)[0]

        imagef = f"{dataDir}/images/{dataType}/{img['file_name']}"
        print('imagef', imagef)             # /home/leo/coco/images/val2017/000000413689.jpg
        print('img_id', img_id)             # 413689
        #print('coco_url', img['coco_url'])  # http://images.cocodataset.org/val2017/000000108503.jpg

        frame = cv2.imread(imagef)
        height_ori, width_ori, _  = frame.shape

        #frame = cv2.flip(frame, 1)

        frame = cv2.resize(frame, comm.YOLO_IMAGE_SIZE)
        height, width, _  = frame.shape

        # load and display instance annotations
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)

        for idx, ann in enumerate(anns):
            print(idx)
            print(' iscrowd', ann['iscrowd'])
            #print(ann)
            if ann['iscrowd']:
                continue

            coco_id = ann['category_id']
            coco_name = comm.COCO_ID_NAME_MAP[coco_id]
            a_box = ann['bbox']
            a_box = [a_box[0]/width_ori, a_box[1]/height_ori, a_box[2]/width_ori, a_box[3]/height_ori]

            info  = [coco_id, coco_name, 1.0, a_box]
            comm.draw_info_on_image(frame, width, height, info, txtfile_cc, 1)

        cv2.imshow('Coco', frame)
        
        if (cv2.waitKey(1000*3) & 0xFF == ord(comm.EXIT_KEY)):
            break

coco_show_categories(coco)

catIds, imgIds = coco_get_image_ids_with_cat(cats = ['person'])

#coco_show_image(catIds, imgIds)
coco_cv_show(catIds, imgIds)