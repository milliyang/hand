#!/bin/bash

COLOR_RED="\e[31;1m"
COLOR_GREEN="\e[32;1m"
COLOR_END="\e[0m"

if [ ! $1 ]; then
    echo -e "${COLOR_GREEN}usage: ${COLOR_END}"
    echo -e "${COLOR_GREEN}  ./run_01_yolo2_darknet_to_hisi.sh 0       # use best   weight ${COLOR_END}"
    echo -e "${COLOR_GREEN}  ./run_01_yolo2_darknet_to_hisi.sh 1       # use latest weight ${COLOR_END}"
    exit 1
fi

LATEST_FILE=$1

# ensure docker running:
docker start darknet2caffe001
docker start caffe111

# caffe docker
    # docker exec -it  darknet2caffe001 /bin/bash

docker exec darknet2caffe001 /root/host/imvt/imvt.caffe/imvt/darknet2caffe/docker_run_yolo2_to_caffe.sh ${LATEST_FILE}
#docker exec darknet2caffe001 /workspace/imvt/imvt.caffe/imvt/darknet2caffe_yolo3/docker_run_yolo2_to_caffe.sh
#output:
#   /home/leo/imvt/imvt.caffe/imvt/darknet2caffe_yolo3


# hisi docker:
    # docker exec -it  caffe111 /bin/bash
    # cd /root/host/imvt/zcam_yolo/hi928docker/yolo
    # python script/transferPic2.py 416

docker exec caffe111 /root/host/imvt/zcam_yolo/hi928docker/yolo/imvt/docker_run_yolo2_416_to_hisi_model.sh
#output:
#   /home/leo/imvt/zcam_yolo/hi928docker/yolo/model_yuv/yolov2_original.om

echo ""

OUT1=/home/leo/imvt/zcam_yolo/hi928docker/yolo/model_yuv/yolov2_original.om
OUT2=/home/leo/imvt/zcam_yolo/zcam_ai_20_y2_s416_v001.om

ls -al ${OUT1}

#cp /home/leo/imvt/zcam_yolo/hi928docker/yolo/model_yuv/yolov2_original.om /home/leo/imvt/zcam_yolo/zcam_ai_20_y2_s416_v001.om
cp ${OUT1} ${OUT2}

md5sum ${OUT1}
md5sum ${OUT2}

echo -e "${COLOR_GREEN} output: ${OUT2} ${COLOR_END}"
