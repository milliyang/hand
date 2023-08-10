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

sudo apt-get -y install libopencv-dev
```


### Build Caffe on Ubuntu
```

mkdir build
cmake ..
make all
make install

```


### hisi
```
cd docker                                   # 包含Dockerfile

docker build -t ss928_image .



```


### Docker Ubuntu
```rust

sudo apt update
sudo apt install apt-transport-https ca-certificates curl gnupg-agent software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io              //install docker


```

### Docker Caffe image:
```

sudo docker pull bvlc/caffe:cpu




```


### run yolo tracker
```

/home/leo/hand/ai/yolo_build

./yolo yolov2.prototxt yolov2.caffemodel street_cars_416x416.jpg

./yolo x_yolov2-voc20.edited.prototxt x_yolov2-voc20.caffemodel leo.png 


```