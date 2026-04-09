import cv2
import mediapipe as mp
import csv
import os

# 初始化 MediaPipe Hand 模块
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# 准备 CSV 数据文件
csv_file = 'gesture_dataset.csv'

# 如果文件不存在，写入表头
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        # 类别标签 + 21个点的 x, y, z 坐标 (共63个特征)
        header = ['label']
        for i in range(21):
            header.extend([f'x_{i}', f'y_{i}', f'z_{i}'])
        writer.writerow(header)

print("启动摄像头，按 '0'-'9' 键录制不同手势的数据。按 'q' 退出。")

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 水平翻转图像，使操作更符合直觉（镜像）
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 进行手部关键点检测
    results = hands.process(rgb_frame)
    
    landmarks_list = []
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # 提取 21 个关键点坐标（相对于手腕点平移，使其具有位置不变性）
            base_x = hand_landmarks.landmark[0].x
            base_y = hand_landmarks.landmark[0].y
            base_z = hand_landmarks.landmark[0].z
            
            for lm in hand_landmarks.landmark:
                # 计算相对坐标
                landmarks_list.extend([
                    lm.x - base_x,
                    lm.y - base_y,
                    lm.z - base_z
                ])
                
    cv2.putText(frame, "Press 0-9 to record, Q to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Data Collection', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif ord('0') <= key <= ord('9'):
        if landmarks_list:
            # 获取按键对应的数字作为标签
            label = chr(key)
            # 保存数据到 CSV
            with open(csv_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                row = [label] + landmarks_list
                writer.writerow(row)
            print(f"Recorded 1 sample for gesture '{label}'")
        else:
            print("No hand detected. Cannot record.")

cap.release()
cv2.destroyAllWindows()
