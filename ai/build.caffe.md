

sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
sudo apt-get install --no-install-recommends libboost-all-dev
sudo apt-get install libopenblas-dev liblapack-dev libatlas-base-dev
sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev
#sudo apt-get install libhdf5-dev


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
