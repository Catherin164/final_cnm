import torch
import torch.nn as nn
import numpy as np
from datetime import datetime, timedelta

class DeliveryForecaster(nn.Module):
    def __init__(self, input_size=10, hidden_size=32, num_layers=2):
        super(DeliveryForecaster, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 16)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(16, 1)  # Dự báo số ngày giao hàng
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

def prepare_delivery_data(delivery_history, seq_length=10):
    """Chuẩn bị dữ liệu giao hàng cho mô hình"""
    sequences = []
    targets = []
    
    for i in range(len(delivery_history) - seq_length):
        seq = delivery_history[i:i + seq_length]
        target = delivery_history[i + seq_length]['delivery_days']
        sequences.append(seq)
        targets.append(target)
    
    return torch.FloatTensor(sequences), torch.FloatTensor(targets)

def extract_features(delivery_record):
    """Trích xuất đặc trưng từ bản ghi giao hàng"""
    features = [
        delivery_record['distance'],  # Khoảng cách
        delivery_record['quantity'],  # Số lượng
        delivery_record['weather_score'],  # Điểm thời tiết (0-1)
        delivery_record['traffic_score'],  # Điểm giao thông (0-1)
        delivery_record['is_weekend'],  # Có phải cuối tuần
        delivery_record['is_holiday'],  # Có phải ngày lễ
        delivery_record['is_rush_hour'],  # Có phải giờ cao điểm
        delivery_record['vehicle_type'],  # Loại phương tiện
        delivery_record['priority'],  # Độ ưu tiên
        delivery_record['previous_delays']  # Số lần trễ trước đó
    ]
    return features

def predict_delivery_time(model, current_features, seq_length=10):
    """Dự báo thời gian giao hàng"""
    model.eval()
    with torch.no_grad():
        # Chuẩn bị dữ liệu đầu vào
        input_seq = torch.FloatTensor(current_features).unsqueeze(0)
        # Dự báo
        prediction = model(input_seq)
        return prediction.item()

def calculate_delivery_metrics(actual_days, predicted_days):
    """Tính toán các chỉ số đánh giá dự báo giao hàng"""
    mae = np.mean(np.abs(actual_days - predicted_days))
    mse = np.mean((actual_days - predicted_days) ** 2)
    rmse = np.sqrt(mse)
    
    # Tính tỷ lệ dự báo đúng trong khoảng ±1 ngày
    within_one_day = np.mean(np.abs(actual_days - predicted_days) <= 1) * 100
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'accuracy_within_one_day': within_one_day
    }

def get_delivery_suggestion(predicted_days, current_stock, daily_consumption):
    """Đưa ra gợi ý về thời gian đặt hàng"""
    days_until_stockout = current_stock / daily_consumption if daily_consumption > 0 else float('inf')
    
    if days_until_stockout <= predicted_days:
        return {
            'suggestion': 'Đặt hàng ngay',
            'urgency': 'high',
            'reason': f'Tồn kho sẽ hết sau {days_until_stockout:.1f} ngày, trong khi thời gian giao hàng dự kiến là {predicted_days:.1f} ngày'
        }
    elif days_until_stockout <= predicted_days + 3:
        return {
            'suggestion': 'Nên đặt hàng sớm',
            'urgency': 'medium',
            'reason': f'Tồn kho sẽ hết sau {days_until_stockout:.1f} ngày, nên đặt hàng trước {predicted_days + 3:.1f} ngày'
        }
    else:
        return {
            'suggestion': 'Có thể đặt hàng sau',
            'urgency': 'low',
            'reason': f'Tồn kho còn đủ cho {days_until_stockout:.1f} ngày, thời gian giao hàng dự kiến là {predicted_days:.1f} ngày'
        }

# Hàm để lưu mô hình
def save_model(model, path):
    """Lưu mô hình"""
    torch.save(model.state_dict(), path)

# Hàm để tải mô hình
def load_model(path, model):
    """Tải mô hình"""
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
