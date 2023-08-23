

### docker gui for amct
```bash


/lhome1/thyang/imvt/zcam_yolo/hi928docker
docker build -t acmt:1804 .
#docker run -d -v $HOME:/root/host -p 22222:22 --name caffe_acmt a2f2977ae003
docker run -it -v /tmp/.X11-unix:/tmp/.X11-unix -v $HOME:/root/host -p 20122:22 --name caffe_acmt a2f2977ae003
docker start caffe_acmt
docker exec -it caffe_acmt /bin/bash
docker rm caffe_acmt



# amct
cd /root/host/imvt/zcam_yolo/hi928docker/Resources/amct_caffe
tar zxvf xxx

cd /root/host/imvt/zcam_yolo/hi928docker/Resources/amct_caffe/caffe_patch
python install.py --caffe_dir /root/host/imvt/imvt.caffe

find /usr/ -name "hotwheels" 

export PYTHONPATH=/root/host/imvt/imvt.caffe/python:$PYTHONPATH

cd /root/host/imvt/zcam_yolo/hi928docker/Resources/amct_caffe/sample/resnet50
cd /root/host/imvt/zcam_yolo/hi928docker/amct_sample/resnet50

#无量化
python3 ./src/ResNet50_sample.py \
--model_file pre_model/ResNet-50-deploy.prototxt \
--weights_file pre_model/ResNet-50-model.caffemodel \
--caffe_dir /root/host/imvt/imvt.caffe/ \
--cpu \
--pre_test 

#量化
python3 ./src/ResNet50_sample.py \
--model_file pre_model/ResNet-50-deploy.prototxt \
--weights_file pre_model/ResNet-50-model.caffemodel \
--caffe_dir /root/host/imvt/imvt.caffe/ \
--cpu 

#转换模型
python3 ./src/convert_model.py \
--model_file pre_model/ResNet-50-deploy.prototxt \
--weights_file pre_model/ResNet-50-model.caffemodel \
--caffe_dir /root/host/imvt/imvt.caffe/ \
--cpu  \
--record_file pre_model/record.txt

```




### docker gui
```bash


/etc/apt/sources.list

deb http://mirrors.aliyun.com/ubuntu/ focal main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ focal main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ focal-security main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ focal-security main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ focal-updates main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ focal-updates main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ focal-proposed main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ focal-proposed main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ focal-backports main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ focal-backports main restricted universe multiverse

sudo apt-get update
#sudo apt-get upgrade

apt install python3.7 curl
apt-get install python3.7-dev
apt install python3-distutils

ln /usr/bin/python3.7 /usr/bin/python -s -f
ln /usr/bin/python3.7 /usr/bin/python3 -s -f

apt install python3-pip
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3.7 get-pip.py


sudo apt install -y build-essential
sudo apt-get install -y xterm cmake git

sudo apt-get install -y firefox xdg-utils
sudo apt-get install -y fonts-droid-fallback fonts-wqy-zenhei fonts-wqy-microhei fonts-arphic-ukai fontsarphic-uming

pip3 install numpy==1.16.0
pip3 install protobuf==3.13.0
pip3 install matplotlib==3.2.0
pip3 install easydict==1.9
pip3 install PyYAML==5.3
pip3 install pillow==6.0.0
pip3 install wget==3.2
pip3 install Cython==0.29.15

sudo apt-get install python3-dev=3.7
pip3 install lmdb==0.98 --use-pep517


pip3 install scikit_image==0.16.2
pip3 install opencv_python==4.2.0.32

/usr/bin/ld: cannot find -lhdf5_hl
/usr/bin/ld: cannot find -lhdf5
/usr/bin/ld: cannot find -lboost_python

```


### xxx
```bash
    # build caffe
    sudo apt install -y build-essential
    sudo apt-get -y install libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler
    sudo apt-get -y install --no-install-recommends libboost-all-dev
    sudo apt-get -y install libopenblas-dev liblapack-dev libatlas-base-dev
    sudo apt-get -y install libgflags-dev libgoogle-glog-dev liblmdb-dev
    sudo apt-get -y install libhdf5-dev
    sudo apt-get -y install libboost-all-dev libboost-python-dev
    sudo apt-get -y install libopencv-dev
```
