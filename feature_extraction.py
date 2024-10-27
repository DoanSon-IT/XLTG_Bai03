# feature_extraction.py
import cv2
import numpy as np

def extract_features(image):
    # Ví dụ trích xuất đặc trưng: color histogram
    histogram = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    histogram = cv2.normalize(histogram, histogram).flatten()
    return histogram
