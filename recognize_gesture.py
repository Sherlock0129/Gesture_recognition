import cv2
import mediapipe as mp
import pickle
import numpy as np
import os

# 加载模型
model_filename = 'gesture_model.pkl'
if not os.path.exists(model_filename):
    print(f"Error: {model_filename} not found. Please train the model first using train_model.py")
    exit()

with open(model_filename, 'rb') as f:
    model = pickle.load(f)

print("Loaded Random Forest Model successfully.")

# 定义手势字典（你可以在此自定义各个数字对应的动作名称）
gesture_names = {
    '0': 'Rock (Stone)',
    '1': 'Paper (Cloth)',
    '2': 'Scissors',
    '3': 'OK',
    '4': 'Thumbs Up',
    '5': 'Thumbs Down'
    # 按照你在 collect_data.py 中录制时的数字含义添加
}

# 初始化 MediaPipe Hand 模块
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)
print("Starting real-time gesture recognition. Press 'q' to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 翻转画面
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # 提取 21 个关键点的相对坐标，必须与训练时完全一致
            landmarks_list = []
            base_x = hand_landmarks.landmark[0].x
            base_y = hand_landmarks.landmark[0].y
            base_z = hand_landmarks.landmark[0].z
            
            for lm in hand_landmarks.landmark:
                landmarks_list.extend([
                    lm.x - base_x,
                    lm.y - base_y,
                    lm.z - base_z
                ])
                
            # 将列表转换为模型可以预测的 2D 数组 (1 行， 63 列)
            features = np.array(landmarks_list).reshape(1, -1)
            
            # 使用训练好的模型预测
            prediction = model.predict(features)[0]
            
            # 映射到对应的手势名称（如果未定义则显示数字编号）
            gesture = gesture_names.get(str(prediction), f"Gesture {prediction}")
            
            # 在视频上显示预测结果
            cv2.putText(frame, f"Gesture: {gesture}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    cv2.imshow('Real-time Gesture Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
