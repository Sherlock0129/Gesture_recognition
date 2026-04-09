import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

def train():
    # 自动选择计算设备: Mac 的 MPS(Apple Silicon), Nvidia 的 CUDA, 或普通 CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU (CUDA)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # 数据集路径
    data_dir = 'dataset/rps'
    if not os.path.exists(data_dir):
        print(f"Error: Dataset directory '{data_dir}' not found.")
        print("Please run 'python download_dataset.py' first.")
        return

    # 定义数据增强和预处理操作
    # MobileNetV2 默认输入尺寸为 224x224，但为了加速训练我们缩小一点
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(), # 随机水平翻转增加数据多样性
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集 (自动按文件夹名作为类别名: paper, rock, scissors)
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    classes = full_dataset.classes
    print(f"Detected classes: {classes}")

    # 划分训练集(80%)和验证集(20%)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 建立模型: 使用轻量级 MobileNetV2 (在移动端/边缘端推理速度极快)
    print("Initializing MobileNetV2 model...")
    # 为了减少下载时间，我们不使用预训练权重，直接从头训练；或者设为 True 使用迁移学习（推荐）
    # 由于这是一个实验，且数据集较小，迁移学习效果最好
    # 在 torchvision 0.13+ 中推荐使用 weights 参数，但为了兼容性保留 pretrained=True
    import warnings
    warnings.filterwarnings("ignore")
    model = models.mobilenet_v2(pretrained=True)
    
    # 替换最后一层全连接层，以适配我们的类别数 (3个类别)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(classes))
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 3 # 仅训练 3 个 epoch 作为演示，防止占用太多时间

    print("\nStarting Training Loop...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        train_acc = 100 * correct_train / total_train
        
        # 验证阶段
        model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        val_acc = 100 * correct_val / total_val
        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Train Loss: {running_loss/len(train_loader):.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss/len(val_loader):.4f} Acc: {val_acc:.2f}%")

    # 保存训练好的模型和类别映射
    model_save_path = 'cnn_gesture_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'classes': classes
    }, model_save_path)
    
    print(f"\nModel training completed! Saved to {model_save_path}")

if __name__ == "__main__":
    train()
