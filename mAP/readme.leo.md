

### Steps:
```bash

# generate model:
    ./run_01_yolo2_darknet_to_hisi.sh 1

# run model generate detection results:

    cd /home/leo/myhome/hand/ai/yolo_build

    ./detect darknet2caffe/imvt20_yolo2.cut.prototxt darknet2caffe/imvt20_yolo2.caffemodel ~/coco/coco_val2017_filelists.txt 1

# 准备数据:
#  - 收集: detection result & ground-truth 结果
    cd /home/leo/myhome/hand/mediapipe

    # 收集, 如:
    #     /home/leo/coco/images/train2017/000000300024.jpg.yolo.txt
    #  -> /home/leo/coco/labels_yolo/train2017/000000300024.jpg.yolo.txt
    #
    python mAP_coco_01_collect.py -filelist ~/coco/coco_val2017_filelists.txt
    #python mAP_coco_01_collect.py -filelist /home/leo/myhome/hand/mediapipe/output/sel_val2017_coco_filelist_all.txt

    # 收集Ground Truth:
    #   "/home/leo/coco/labels/val2017/xxxxx.txt =>
    #   "/home/leo/coco/val_gt_labels/val2017/xxxxx.txt
    # 收集Yolo检测结果:
    #   "/home/leo/coco/labels_yolo/val2017/xxxxx.txt =>
    #   "/home/leo/coco/val_yolo_labels/val2017/xxxxx.txt
    python mAP_coco_03_chk_label_data.py

# calc mAP:
#   input:  /home/leo/myhome/hand/mAP/input             (symbol link to ground  truth)
#   output: /home/leo/myhome/hand/mAP/output
    cd /home/leo/myhome/hand/mAP
    python main.py -na


```