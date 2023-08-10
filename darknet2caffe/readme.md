
### Setup Docker on Ubuntu
```rust

sudo apt update
sudo apt install apt-transport-https ca-certificates curl gnupg-agent software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io              //install docker



//update source mirror (faster)
sudo vim /etc/docker/daemon.json

{
    "registry-mirrors": ["http://hub-mirror.c.163.com"]
}

sudo systemctl restart docker.service
sudo docker info
```

### Docker for not root user  (not safe)
```
sudo groupadd docker
sudo usermod -aG docker leo
newgrp docker
docker version

```


### Docker pull Caffe image:
```rust
*maybe sudo*

docker pull bvlc/caffe:cpu                     //pull image
docker images                                  //show docker images

docker run -it -v $PWD:/workspace -p 20022:22 --name darknet2caffe001 0b577b836386



docker start darknet2caffe001
docker exec -it  darknet2caffe001 /bin/bash
docker stop darknet2caffe001

docker rm darknet2caffe001

```


### Convert Darknet To Caffe in (darknet2caffe001)
```rust

pip install torch
pip install future

cd /workspace/github/darknet2caffe
git checkout master

python darknet2caffe.py x_yolov2-voc20.cfg x_yolov2-voc20.weight


```
