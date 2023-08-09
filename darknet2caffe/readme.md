
### Setup Docker on Ubuntu
```rust

sudo apt update
sudo apt install apt-transport-https ca-certificates curl gnupg-agent software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io              //install docker


```


### Docker pull Caffe image:
```rust

sudo docker pull bvlc/caffe:cpu             //pull image
docker images                               //show docker images


docker run -it -v $PWD:/root/host -p 20022:22 --name darknet2caffe001 a2f2977ae003

docker exec -it  darknet2caffe001 /bin/bash



```


