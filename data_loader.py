# data_loader.py
import cv2
import os

def load_images_from_folder(folder, label, X, y):
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (100, 100))  # Chỉnh kích thước ảnh
            X.append(img)
            y.append(label)
