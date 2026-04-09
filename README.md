# 基于计算机视觉的手势识别 (IOT Lab 1)

本项目提供了**两种**手势识别的实现方案，以满足 IOT Lab 的要求：

1. **方案 A：MediaPipe 关键点特征提取 + 随机森林 (超轻量级)**
2. **方案 B：PyTorch 深度学习 CNN 图像分类 (经典数据集训练法)** —— *推荐*

## 目录结构
- `plan.md` - 项目实施计划和思路
- `requirements.txt` - Python 依赖列表
- **[方案 A 文件]**:
  - `collect_data.py` - 数据采集与特征提取脚本 (手动录制少量数据集)
  - `train_model.py` - 模型训练与验证脚本 (Random Forest)
  - `recognize_gesture.py` - 实时手势识别推理脚本 (加载 .pkl)
- **[方案 B 文件]**:
  - `download_dataset.py` - 下载公共小型数据集 (Rock-Paper-Scissors)
  - `train_cnn.py` - 深度学习 CNN 图像分类模型训练脚本
  - `recognize_gesture_cnn.py` - 基于 PyTorch 模型的实时摄像头推理

## 环境搭建
请确保你的电脑上安装了 Python 3 (推荐使用虚拟环境)。
1. 在终端中安装所需的依赖：
```bash
pip install -r requirements.txt
```
*(注意：方案 B 引入了 PyTorch，如果在 M1/M2/M3 Mac 上运行，PyTorch 会自动调用 MPS 进行 GPU 加速。)*

---

## 方案 B：使用图像数据集训练 CNN 模型 (推荐)

如果您希望使用标准的深度学习流程，即**“下载公开数据集 -> 构建 CNN 网络 -> 训练 -> 部署推理”**，请使用此方案。

### 步骤 1：下载并准备数据集
我们使用 Laurence Moroney 提供的经典“石头-剪刀-布”公开数据集，该数据集体积较小（约 200MB），适合快速实验：
```bash
python download_dataset.py
```
这将在当前目录下创建 `dataset/rps/` 文件夹，包含 `rock`、`paper` 和 `scissors` 三个类别的图片。

### 步骤 2：训练 MobileNetV2 模型
利用下载好的图片数据集训练一个轻量级的卷积神经网络（MobileNetV2）。
```bash
python train_cnn.py
```
- 脚本会自动划分 80% 作为训练集，20% 作为验证集。
- 自动利用 GPU (CUDA 或 Mac Apple Silicon MPS) 加速训练。
- 训练完成后，模型将被保存为 `cnn_gesture_model.pth`。

### 步骤 3：实时手势推理
使用刚才训练好的 CNN 模型对摄像头的实时视频流进行分类预测：
```bash
python recognize_gesture_cnn.py
```
- 对着镜头展示“石头”、“剪刀”或“布”，画面上方将显示模型预测结果及置信度。
- 按键盘 `q` 退出。

---

## 方案 A：手动采集特征点训练 (无需大数据集)

如果您希望模型计算量极小，并想自己定义手势类型（如 OK、点赞等），可使用方案 A。

1. **录制特征数据：** 运行 `python collect_data.py` 启动摄像头。对着摄像头做出手势，按数字键 `0`-`9` 将当前手势的特征保存到 `gesture_dataset.csv` 中。
2. **训练模型：** 运行 `python train_model.py` 训练一个基于特征坐标的随机森林模型，生成 `gesture_model.pkl`。
3. **实时推理：** 运行 `python recognize_gesture.py` 启动摄像头进行识别。
