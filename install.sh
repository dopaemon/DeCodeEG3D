#/bin/bash

export DEBIAN_FRONTEND=noninteractive

nvidiaverin = 381

sudo apt purge nvidia-*
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
sudo apt install -yq nvidia-$nvidiaverin git ninja-build

sudo apt install python-is-python3 python3-pip -y

git clone https://github.com/NVlabs/eg3d

model_url = "https://api.ngc.nvidia.com/v2/models/nvidia/research/eg3d/versions/1/files/ffhq512-128.pkl"  #@param {type: "string"}

wget -c $model_url -O model.pkl

pip install scikit-video
pip install lpips
pip install trimesh

