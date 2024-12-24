import cv2
import numpy as np
from keras.preprocessing import image # type: ignore
from keras.models import load_model # type: ignore
import os

# Khởi tạo kết nối với camera
vid = cv2.VideoCapture(0)
print("Camera connection successfully established")

# Khởi tạo biến đếm số lượng ảnh đã chụp
i = 0

# Danh sách các lớp mà mô hình sẽ dự đoán
classes = ['Chuối','Dâu tây','Dứa','Khế','Xoài']

# Tải mô hình máy học đã được đào tạo từ tệp 'my_model.keras'
new_model = load_model('my_model.keras')

# Bắt đầu vòng lặp vô hạn để liên tục chụp và xử lý các khung hình từ camera
while(True):
    # Đọc một khung hình từ camera
    r, frame = vid.read()
    
    # Hiển thị khung hình lên màn hình
    cv2.imshow('frame', frame)
    
    # Lưu khung hình hiện tại vào một tệp ảnh với tên tăng dần
    cv2.imwrite(r'D:\Do_anst2\csudai\\final' + str(i) + ".jpg", frame)
    
    # Tải ảnh từ tệp vừa lưu và thay đổi kích thước của ảnh thành 64x64 pixels
    test_image = image.load_img(r'D:\Do_anst2\csudai\\final' + str(i) + ".jpg", target_size=(64, 64))
    
    # Chuyển ảnh thành mảng numpy
    test_image = image.img_to_array(test_image)
    
    # Mở rộng số chiều của mảng numpy để phù hợp với đầu vào của mô hình
    test_image = np.expand_dims(test_image, axis=0)
    
    # Dự đoán lớp của ảnh
    result = new_model.predict(test_image)
    
    # Lấy kết quả dự đoán đầu tiên
    result1 = result[0]
    
    # Duyệt qua các lớp trong danh sách 'classes' để tìm lớp có xác suất dự đoán cao nhất
    for y in range(5):
        if result1[y] == 1.:
            break
    
    # Lấy tên của lớp được dự đoán từ danh sách 'classes'
    prediction = classes[y]
    
    # Hiển thị tên của lớp được dự đoán
    print(prediction)
    
    # Xóa tệp ảnh vừa lưu sau khi đã sử dụng
    os.remove(r'D:\Do_anst2\csudai\\final' + str(i) + ".jpg")
    
    # Tăng biến đếm để đảm bảo tên của các tệp ảnh mới không trùng lặp
    i = i + 1
    
    # Đợi một phím được nhấn, nếu phím là 'q' thì thoát khỏi vòng lặp
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên camera
vid.release()

# Đóng tất cả các cửa sổ hiển thị hình ảnh của OpenCV
cv2.destroyAllWindows()
