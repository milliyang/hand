#!/bin/bash


# caffe docker
    # docker exec -it  darknet2caffe001 /bin/bash

docker exec darknet2caffe001 /workspace/imvt/imvt.caffe/imvt/darknet2caffe/docker_run_yolo2_to_caffe.sh
#output:
#   /home/leo/imvt/imvt.caffe/imvt/darknet2caffe_yolo3


# hisi docker:
    # docker exec -it  caffe111 /bin/bash
    # cd /root/host/imvt/zcam_yolo/hi928docker/yolo
    # python script/transferPic3.py 416

docker exec caffe111 /root/host/imvt/zcam_yolo/hi928docker/yolo/imvt/docker_run_yolo2_416_to_hisi_model.sh
#output:
#   /home/leo/imvt/zcam_yolo/hi928docker/yolo/model_yuv/yolov2_original.om

ls -al /home/leo/imvt/zcam_yolo/hi928docker/yolo/model_yuv/yolov2_original.om

mv /home/leo/imvt/zcam_yolo/hi928docker/yolo/model_yuv/yolov2_original.om   /home/leo/imvt/zcam_yolo/zcam_ai_20_y2_s416_v001.om