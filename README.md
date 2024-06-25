# Aimbot：基于yolov5的csgo游戏自瞄外挂

## 项目简介

本项目使用yolov5，训练了427张csgo的游戏截图，实现了csgo的自动瞄准、自动射击功能。

识别目标位置后，能够聚焦到目标周边的小范围进一步识别，对于小目标具有较好的识别效果、并具备较高性能。

推理采用CPU，硬件要求较低，适用面广。

## requirements

 `pip install -r requirements.txt`

python>=3.7

matplotlib>=3.2.2

numpy>=1.18.5

torch>=1.7.0

torchvision>=0.8.1

**更多依赖信息：requirements.txt**

## 模型训练

数据集：训练集337张图片，验证集90张图片

训练参数：epoch 100，batch_size 16，img_size 640

## 训练结果与性能评估

训练结果：权重文件`weights/best.pt`

验证集上的precision（查准率）0。93，recall（查全率）0.836，mAP_0.5（阈值为0.5时计算得到的平均精度）：0.871

训练过程中loss和相关metric的变化：

![results](https://github.com/ttttkx/CSGO_Aimbot/assets/144672418/fe9a4314-e40e-4e44-8f6d-b195402b1d74)

最终模型在验证集上的可视化结果：

![val_batch2_labels](https://github.com/ttttkx/CSGO_Aimbot/assets/144672418/e38a7497-9fd0-40e3-8ca8-e3ed086c9182)



