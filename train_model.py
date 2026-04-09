import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import os

# 确保数据集存在
csv_file = 'gesture_dataset.csv'
if not os.path.exists(csv_file):
    print(f"Error: {csv_file} not found. Please run collect_data.py first.")
    exit()

# 加载数据
data = pd.read_csv(csv_file)

# X 是特征 (21个关键点的相对 x,y,z 坐标)，y 是标签 (手势编号 0-9)
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用随机森林分类器，这是一个轻量级且高效的模型
print("Training Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 评估模型准确率
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on Test Data: {accuracy * 100:.2f}%")

# 保存模型为 .pkl 文件
model_filename = 'gesture_model.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(model, f)
    
print(f"Model successfully saved to {model_filename}")
