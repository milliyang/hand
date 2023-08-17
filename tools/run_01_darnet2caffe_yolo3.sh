#!/bin/bash


docker exec darknet2caffe001 /workspace/imvt/imvt.caffe/imvt/darknet2caffe_yolo3/docker_run_yolo3_to_caffe.sh
#output:
#   /home/leo/imvt/imvt.caffe/imvt/darknet2caffe_yolo3


docker exec caffe111 /root/host/imvt/zcam_yolo/hi928docker/yolo/imvt/docker_run_yolo3_320_to_hisi_model.sh
#output:
#   /home/leo/imvt/zcam_yolo/hi928docker/yolo/model_yuv/yolov3_original.om
