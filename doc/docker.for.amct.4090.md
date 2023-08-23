
### docker gui for amct
```bash

cd /home/leo/imvt/zcam_yolo/hi928docker
docker build -t acmt:1804 .
#docker run -d -v $HOME:/root/host -p 22222:22 --name caffe_acmt e6b8433cbb28
docker run -it -v /tmp/.X11-unix:/tmp/.X11-unix -v $HOME:/root/host -p 20122:22 --name caffe_acmt e6b8433cbb28
docker start caffe_acmt
docker exec -it caffe_acmt /bin/bash

#docker rm caffe_acmt


# amct
cd /root/host/imvt/zcam_yolo/hi928docker/Resources/amct_caffe
tar zxvf amct_caffe_sample.tar.gz
tar zxvf caffe_patch.tar.gz

#cd /root/host/imvt/zcam_yolo/hi928docker/Resources/amct_caffe/caffe_patch
#python install.py --caffe_dir /root/host/imvt/imvt.caffe.amct

find /usr/ -name "hotwheels" 


export PYTHONPATH=/root/host/imvt/imvt.caffe.amct/python:$PYTHONPATH

cd /root/host/imvt/zcam_yolo/hi928docker/amct_sample/resnet50

#protobuf error, downgrade version:
pip3 install protobuf==3.13.0

#无量化
python3 ./src/ResNet50_sample.py \
--model_file pre_model/ResNet-50-deploy.prototxt \
--weights_file pre_model/ResNet-50-model.caffemodel \
--caffe_dir /root/host/imvt/imvt.caffe.amct/ \
--cpu \
--pre_test 

#量化
python3 ./src/ResNet50_sample.py \
--model_file pre_model/ResNet-50-deploy.prototxt \
--weights_file pre_model/ResNet-50-model.caffemodel \
--caffe_dir /root/host/imvt/imvt.caffe.amct/ \
--cpu 

#转换模型
python3 ./src/convert_model.py \
--model_file pre_model/ResNet-50-deploy.prototxt \
--weights_file pre_model/ResNet-50-model.caffemodel \
--caffe_dir /root/host/imvt/imvt.caffe.amct/ \
--cpu  \
--record_file pre_model/record.txt



###

docker build -t gui1804 .

docker run -d -v $HOME:/root/host --cap-add=SYS_PTRACE --shm-size=2048m -p 22222:22 -p 14000:4000  --name caffegui001 b6ef9fb105

docker exec -it caffegui001 /bin/bash

docker rm caffegui001



#fast check
sudo apt-cache policy docker-ce docker-ce-cli


apt-get autoremove docker docker-ce docker-engine  docker.io  containerd runc
apt-get autoremove docker-ce-*

sudo apt-get install docker-ce=5:23.0.0-1~ubuntu.18.04~bionic  docker-ce-cli=5:23.0.0-1~ubuntu.18.04~bionic

docker pull ubuntu:18.04
sudo docker run -i -t ubuntu:18.04 /bin/bash


#docker system prune -a --force

```
