Đoàn Văn Sơn 20211512
Ý nghĩa các Folder/File dùng trong bài.
Folder: Horse và Lion là 2 folder chứa hình ảnh để đưa vào huấn luyện, mỗi folder có 100 ảnh
File: data_loader.py.
- Mục đích  Chứa hàm load_images_from_folder dùng để tải và chuẩn hóa hình ảnh từ một thư mục.
File: feature_extraction.py.
- Mục đích: Chứa hàm extract_features để trích xuất các đặc trưng của hình ảnh.
File: train_model.py.
- Mục đích: Dùng để huấn luyện và lưu trữ mô hình KNN. với k =5, sau đó lưu dữ liệu ra knn_model.pkl
File: classify_image.py.
- Mục đích: Dùng để phân loại một ảnh mới dựa trên mô hình đã huấn luyện.
File: check_model_v2.py.
- Mục đích ban đầu: dùng để kiểm tra xem mã KNN có hoạt động bình thường hay không vì VSCode không hỗ trợ.
File: them_du_lieu.py.
- Mục đích ban đầu: dùng để tạo thêm các bản sao nhưng chỉnh ảnh quay các hướng để có tính đa dạng về dữ liệu huấn luyện
nhưng cho nhiều quá lại làm giảm độ chính xác khi dự đoán.
Các file img test1--> test10 là các ảnh để chúng ta thêm vào nhằm kiểm tra dự đoán xem đó là Lion hay Horse.
KẾT QUẢ: độ chính xác trên 70%. Tùy thuộc vào chất lượng ảnh, độ nhiễu và cách chúng ta chọn K thì dự đoán sẽ cao.
