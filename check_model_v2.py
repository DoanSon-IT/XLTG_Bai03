import joblib

# Tải mô hình từ file
model = joblib.load('knn_model.pkl')

# In ra thông tin mô hình
print("Thong tin mo hinh KNN:")
print(model)
print("So lang gieng:", model.n_neighbors)
print("Metric:", model.metric)
