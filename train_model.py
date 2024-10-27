# train_model.py
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from data_loader import load_images_from_folder
from feature_extraction import extract_features

# Tạo dataset
X = []
y = []

load_images_from_folder('./Lion', 1, X, y)  # Label 1 cho Lion
load_images_from_folder('./Horse', 0, X, y)  # Label 0 cho Horse

# Trích xuất đặc trưng cho từng ảnh
X_features = [extract_features(img) for img in X]

X = np.array(X_features)
y = np.array(y)

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Huấn luyện mô hình KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Lưu mô hình
joblib.dump(knn, 'knn_model.pkl')

# Đánh giá mô hình
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print("Do chinh xac:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
