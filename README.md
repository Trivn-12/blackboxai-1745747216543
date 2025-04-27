# Image Recognition App

Ứng dụng web đơn giản sử dụng mô hình MobileNet đã được huấn luyện sẵn để nhận diện hình ảnh.

## Tính năng

- Tải ảnh lên qua giao diện web
- Backend xử lý ảnh và nhận diện bằng mô hình MobileNet
- Hiển thị 3 nhãn dự đoán hàng đầu cùng xác suất

## Cài đặt

1. Tạo môi trường ảo (khuyến nghị):
```
python3 -m venv venv
source venv/bin/activate
```

2. Cài đặt các thư viện:
```
pip install -r requirements.txt
```

## Chạy ứng dụng

```
python app.py
```

Sau đó mở trình duyệt và truy cập http://127.0.0.1:5000 để sử dụng ứng dụng.

## Ghi chú

- Ứng dụng sử dụng Flask làm backend và TensorFlow MobileNet để nhận diện ảnh.
- Frontend sử dụng Tailwind CSS và Font Awesome để tạo giao diện hiện đại, thân thiện.
