### ubuntu
```rust

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
sudo apt-get upgrade


sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler
sudo apt-get install --no-install-recommends libboost-all-dev
sudo apt-get install libopenblas-dev liblapack-dev libatlas-base-dev
sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev
sudo apt-get install libhdf5-dev

sudo apt-get -y install libboost-all-dev
sudo apt-get -y install libboost-python-dev

sudo apt-get -y install libopencv-dev=3.4
```

### Build Caffe on Ubuntu
```bash

## check opencv:
export PKG_CONFIG_PATH=/home/leo/myhome/download/opencv/install/lib/pkgconfig:$PKG_CONFIG_PATH

pkg-config --cflags opencv; pkg-config --libs opencv; pkg-config opencv --modversion

mkdir build4090; cd build4090
cmake ..
make all
make install

#install: /home/leo/imvt/imvt.caffe/build4090/install

```


### run yolo tracker
```bash

/home/leo/hand/ai/yolo_build

./yolo yolov2.prototxt yolov2.caffemodel street_cars_416x416.jpg

./yolo x_yolov2-voc20.edited.prototxt x_yolov2-voc20.caffemodel leo.png 

./yolo darknet2caffe/imvt20_yolo2.cut.prototxt darknet2caffe_yolo3/imvt20_yolo2.caffemodel leo.png
./yolo darknet2caffe/imvt20_yolo2.cut.prototxt darknet2caffe/imvt20_yolo2.caffemodel leo.png

```