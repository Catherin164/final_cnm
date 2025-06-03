import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
from datetime import datetime, timedelta

# Hàm dự báo nhu cầu cho từng sản phẩm dựa trên lịch sử sales_volume
# sales_history: list các số lượng bán ra theo từng tháng (hoặc tuần)
# periods_ahead: số kỳ muốn dự báo (mặc định: 1 kỳ tiếp theo)
def forecast_demand(sales_history, periods_ahead=1):
    if not sales_history or len(sales_history) < 3:
        # Không đủ dữ liệu để dự báo
        return None
    X = np.arange(len(sales_history)).reshape(-1, 1)
    y = np.array(sales_history)
    model = LinearRegression()
    model.fit(X, y)
    future_X = np.arange(len(sales_history), len(sales_history) + periods_ahead).reshape(-1, 1)
    forecast = model.predict(future_X)
    return int(max(0, round(forecast[-1])))

# Hàm tạo dữ liệu lịch sử mẫu từ sales_volume tổng (giả sử chia đều cho 12 tháng)
def generate_monthly_history(total_sales, months=12):
    if pd.isnull(total_sales) or total_sales == 0:
        return [0] * months
    avg = int(total_sales // months)
    history = [avg] * months
    # Phân bổ phần dư cho các tháng cuối
    for i in range(total_sales % months):
        history[-(i+1)] += 1
    return history 