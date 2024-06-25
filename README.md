# Aimbot：基于yolov5的csgo游戏自瞄外挂

## 项目简介

本项目使用yolov5，训练了300张csgo的游戏截图，实现了csgo的自动瞄准、自动射击功能。

识别目标位置后，聚焦到目标周边的小范围进一步识别，对于小目标具有较好的识别效果、并具备较高性能。

推理采用CPU，硬件要求较低，适用面广。

## requirements

 `pip install -r requirements.txt`

gitpython

ipython

matplotlib>=3.2.2

numpy>=1.18.5

opencv-python>=4.1.1

Pillow>=7.1.2

psutil

PyYAML>=5.3.1

requests>=2.23.0

scipy>=1.4.1

thop>=0.1.1

torch>=1.7.0

torchvision>=0.8.1

tqdm>=4.64.0



