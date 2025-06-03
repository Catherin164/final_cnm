# Hệ thống Quản lý Kho và Bán hàng

Hệ thống quản lý kho và bán hàng thông minh với các tính năng:
- Quản lý hàng hóa (nhập, xuất, hoàn trả)
- Dự báo nhu cầu
- Truy xuất nguồn gốc
- Giao hàng thông minh
- Hợp đồng thông minh
- Báo cáo & phân tích
- Quản lý tài khoản & phân quyền

## Yêu cầu hệ thống

- Python 3.8 trở lên
- pip (Python package manager)

## Cài đặt

1. Clone repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Tạo môi trường ảo và kích hoạt:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

4. Khởi tạo cơ sở dữ liệu và import dữ liệu mẫu:
```bash
python import_data.py
```

5. Chạy ứng dụng:
```bash
python app.py
```

6. Truy cập ứng dụng tại: http://localhost:5000

## Thông tin đăng nhập mặc định

- Username: admin
- Password: admin123

## Cấu trúc thư mục

```
.
├── app.py              # File chính của ứng dụng
├── import_data.py      # Script import dữ liệu
├── requirements.txt    # Danh sách thư viện
├── templates/          # Thư mục chứa templates
│   ├── base.html      # Template cơ sở
│   ├── index.html     # Trang chủ
│   ├── inventory.html # Quản lý hàng hóa
│   └── login.html     # Trang đăng nhập
└── static/            # Thư mục chứa file tĩnh
```

## Tính năng chính

1. **Trang chủ**
   - Thống kê tổng quan
   - Cảnh báo kho
   - Biểu đồ doanh số

2. **Quản lý hàng hóa**
   - Nhập hàng
   - Xuất hàng
   - Hoàn trả
   - Quản lý tồn kho

3. **Dự báo nhu cầu**
   - Phân tích xu hướng
   - Đề xuất nhập hàng
   - Tối ưu tồn kho

4. **Truy xuất nguồn gốc**
   - Lịch sử lô hàng
   - Thông tin nhà cung cấp
   - Chứng nhận chất lượng

5. **Giao hàng thông minh**
   - Lộ trình tối ưu
   - Quản lý đơn hàng
   - Theo dõi vận chuyển

6. **Hợp đồng thông minh**
   - Quản lý điều kiện
   - Tự động hóa giao dịch
   - Theo dõi thực hiện

7. **Báo cáo & phân tích**
   - Dashboard realtime
   - Báo cáo chi tiết
   - Phân tích xu hướng

8. **Tài khoản & phân quyền**
   - Quản lý người dùng
   - Phân quyền truy cập
   - Ghi log hoạt động

## Đóng góp

Mọi đóng góp đều được hoan nghênh. Vui lòng tạo issue hoặc pull request để đóng góp.

## Giấy phép

MIT License 