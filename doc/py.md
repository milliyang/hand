
### ubuntu python3
```
sudo rm /usr/bin/python
sudo ln -s /usr/bin/python3  /usr/bin/python

sudo apt-get install python3-pip

pip install numpy pandas omegaconf tqdm
```

### Setup Mediapipe Env:
```
conda remove --name hands --all
conda create -n hands python=3.9
conda activate hands
conda install swig

cd mediapipe/
pip install -r requirements.txt


```



