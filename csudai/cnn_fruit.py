import numpy as np
from keras_preprocessing import image
import cv2
import os
import tensorflow as tf
import time
from keras_preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore

# Tiền xử lý dữ liệu: thiết lập các generator dữ liệu với augmentation cho tập huấn luyện
train_datagen = ImageDataGenerator(
    rescale=1./255,             # Chia lại giá trị pixel trong khoảng [0,1]
    shear_range=0.2,            # Biến đổi cắt
    zoom_range=0.2,             # Biến đổi zoom
    horizontal_flip=True        # Lật ảnh theo chiều ngang
)

# Lưu lượng hình ảnh huấn luyện theo batch kích thước 12 sử dụng generator train_datagen
training_set = train_datagen.flow_from_directory(
    'D:\\Do_anst2\\csudai\\train1',       # Đường dẫn đến các ảnh huấn luyện
    target_size=(64, 64),       # Thay đổi kích thước ảnh thành 64x64
    batch_size=12,              # Kích thước batch
    class_mode='categorical'    # Chế độ lớp cho các nhãn phân loại
)

# Tiền xử lý dữ liệu: thiết lập generator dữ liệu không có augmentation cho tập kiểm tra
test_datagen = ImageDataGenerator(rescale=1./255)

# Lưu lượng hình ảnh kiểm tra theo batch kích thước 12 sử dụng generator test_datagen
test_set = test_datagen.flow_from_directory(
    'D:\Do_anst2\csudai\\test1',        # Đường dẫn đến các ảnh kiểm tra
    target_size=(64, 64),       # Thay đổi kích thước ảnh thành 64x64
    batch_size=12,              # Kích thước batch
    class_mode='categorical'    # Chế độ lớp cho các nhãn phân loại
)

# Xác định các lớp cho việc phân loại
classes = ['Chuối', 'Dâu tây', 'Dứa', 'Khế', 'Xoài']

# In thông điệp trạng thái
print("Đang xử lý ảnh.......Hoàn tất")

# Xây dựng mô hình CNN
cnn = tf.keras.models.Sequential()

# In thông điệp trạng thái
print("Đang xây dựng mạng Neural.....")

# Thêm lớp tích chập đầu tiên với activation ReLU
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

# Thêm lớp max pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Thêm lớp tích chập thứ hai với activation ReLU
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))

# Thêm lớp max pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Làm phẳng các feature map
cnn.add(tf.keras.layers.Flatten())

# Thêm các lớp kết nối đầy đủ với activation ReLU
cnn.add(tf.keras.layers.Dense(units=32, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=64, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=256, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=256, activation='relu'))

# Thêm lớp đầu ra với activation softmax cho việc phân loại
cnn.add(tf.keras.layers.Dense(units=5, activation='softmax'))

# Biên dịch mô hình với optimizer Adam và loss categorical cross-entropy
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# In thông điệp trạng thái
print("Đang huấn luyện cnn")

# Huấn luyện mô hình với dữ liệu huấn luyện và kiểm tra
cnn.fit(x=training_set, validation_data=test_set, epochs=15)

# Lưu mô hình đã huấn luyện
cnn.save("my_model.keras")
