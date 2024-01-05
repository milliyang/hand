

### Steps:
```bash

# generate model:
    ./run_01_yolo2_darknet_to_hisi.sh 1


# run model generate detection results:

    cd /home/leo/myhome/hand/ai/yolo_build

    ./detect darknet2caffe/imvt20_yolo2.cut.prototxt darknet2caffe/imvt20_yolo2.caffemodel ~/coco/coco_val2017_filelists.txt 1

# prepare data
    cd /home/leo/myhome/hand/mediapipe
    python mAP_coco_01_collect.py
    python mAP_coco_03_chk_label_data.py


# calc mAP:
    cd /home/leo/myhome/hand/mAP

    python main.py -na



```