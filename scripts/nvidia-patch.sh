set -ex

git clone --depth 1 https://github.com/keylase/nvidia-patch.git

cd nvidia-patch

sudo bash ./patch.sh
