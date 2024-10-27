# classify_image.py
import numpy as np
import joblib
import cv2
from feature_extraction import extract_features

# Tải mô hình
knn = joblib.load('knn_model.pkl')

# Hàm phân loại ảnh mới
def classify_new_image(image_path):
    img = cv2.imread(image_path)
    if img is not None:
        features = extract_features(img)
        features = features.reshape(1, -1)
        prediction = knn.predict(features)[0]
        label = "Lion" if prediction == 1 else "Horse"
        print(f"Du doan anh nay la: {label}")
    else:
        print("Khong the doc anh.")

# Thay đổi đường dẫn ảnh tại đây
image_path = './test6.jpg'  
classify_new_image(image_path)
