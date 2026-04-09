import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

def load_model(model_path, device):
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        print("Please run 'python train_cnn.py' first.")
        return None, None
        
    print(f"Loading trained CNN model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    classes = checkpoint['classes']
    
    # 重建相同的 MobileNetV2 网络结构
    model = models.mobilenet_v2(pretrained=False)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(classes))
    
    # 载入训练好的权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval() # 切换为推理模式
    
    return model, classes

def recognize_realtime():
    # 自动选择计算设备
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    model_path = 'cnn_gesture_model.pth'
    model, classes = load_model(model_path, device)
    
    if model is None:
        return

    # 定义与训练时一致的图像预处理 (无需数据增强)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("\n[CNN] Starting real-time gesture recognition. Press 'q' to exit.")
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 翻转画面使其像镜子一样，更符合用户直觉
        frame = cv2.flip(frame, 1)
        
        # 将 BGR (OpenCV格式) 转换为 RGB (PyTorch/PIL格式)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # 预处理图像：调整大小、转 Tensor、归一化，然后增加 batch 维度 (1, C, H, W)
        input_tensor = transform(pil_image).unsqueeze(0).to(device)
        
        # 进行推理
        with torch.no_grad():
            outputs = model(input_tensor)
            # 获取最高概率的索引
            _, predicted_idx = torch.max(outputs, 1)
            
            # 使用 Softmax 获取置信度 (Confidence)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence = probabilities[0][predicted_idx].item()
            
            predicted_class = classes[predicted_idx.item()]
            
        # 只有在置信度较高时才认为是有效手势
        if confidence > 0.6:
            text = f"Gesture: {predicted_class.upper()} ({confidence*100:.1f}%)"
            color = (0, 255, 0) # 绿色
        else:
            text = "Gesture: UNKNOWN"
            color = (0, 0, 255) # 红色

        # 在画面上绘制文字
        cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        cv2.imshow('CNN Gesture Recognition (Deep Learning)', frame)
        
        # 监听按键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_realtime()
