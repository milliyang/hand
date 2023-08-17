#!/bin/bash

# caffe docker
    # docker exec -it  darknet2caffe001 /bin/bash

docker exec darknet2caffe001 /workspace/imvt/imvt.caffe/imvt/darknet2caffe_yolo3/docker_run_yolo3_to_caffe.sh
#output:
#   /home/leo/imvt/imvt.caffe/imvt/darknet2caffe_yolo3


# hisi docker:
    # docker exec -it  caffe111 /bin/bash
    # cd /root/host/imvt/zcam_yolo/hi928docker/yolo
    # python script/transferPic3.py 320

docker exec caffe111 /root/host/imvt/zcam_yolo/hi928docker/yolo/imvt/docker_run_yolo3_320_to_hisi_model.sh
#output:
#   /home/leo/imvt/zcam_yolo/hi928docker/yolo/model_yuv/yolov3_original.om

ls -al /home/leo/imvt/zcam_yolo/hi928docker/yolo/model_yuv/yolov3_original.om

cp /home/leo/imvt/zcam_yolo/hi928docker/yolo/model_yuv/yolov3_original.om   /home/leo/imvt/zcam_yolo/zcam_ai_20_y3_s320_v001.om
