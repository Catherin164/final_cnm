import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class ConsumptionForecaster(nn.Module):
    def __init__(self, input_size=30, hidden_size=64, num_layers=2, output_size=7):
        super(ConsumptionForecaster, self).__init__()
        
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
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, output_size)
        
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

def load_reorder_levels(csv_path):
    """Load reorder levels from CSV file"""
    try:
        df = pd.read_csv(csv_path)
        return dict(zip(df['Product_ID'], df['Reorder_Level']))
    except Exception as e:
        print(f"Error loading reorder levels: {e}")
        return {}

def prepare_data(data, seq_length=30):
    """Prepare data for the model with seasonal adjustments"""
    sequences = []
    targets = []
    
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        target = data[i + seq_length:i + seq_length + 7]
        
        # Add seasonal adjustments
        day_of_week = (i + seq_length) % 7
        if day_of_week in [5, 6]:  # Weekend
            target = target * 1.2  # 20% increase on weekends
        elif day_of_week == 0:  # Monday
            target = target * 0.9  # 10% decrease on Mondays
            
        sequences.append(seq)
        targets.append(target)
    
    return torch.FloatTensor(sequences), torch.FloatTensor(targets)

def analyze_consumption_pattern(historical_data, window=30):
    """Analyze consumption patterns for more realistic predictions"""
    patterns = {
        'daily_avg': np.mean(historical_data[-window:]),
        'daily_std': np.std(historical_data[-window:]),
        'weekly_pattern': np.array([np.mean(historical_data[-window:][i::7]) for i in range(7)]),
        'trend': np.polyfit(range(window), historical_data[-window:], 1)[0]
    }
    return patterns

def adjust_prediction(prediction, patterns):
    """Adjust prediction based on patterns and seasonality"""
    adjusted = prediction.copy()
    
    # Apply weekly pattern
    for i in range(len(prediction)):
        day_of_week = i % 7
        adjusted[i] *= patterns['weekly_pattern'][day_of_week] / patterns['daily_avg']
    
    # Apply trend
    trend_factor = 1 + patterns['trend'] * np.arange(len(prediction)) / patterns['daily_avg']
    adjusted *= trend_factor
    
    # Add random variation within historical standard deviation
    noise = np.random.normal(0, patterns['daily_std'] * 0.1, len(prediction))
    adjusted += noise
    
    return np.maximum(adjusted, 0)  # Ensure non-negative values

def check_reorder_status(current_stock, predicted_consumption, reorder_level, safety_stock=0.2):
    """Check if reorder is needed based on current stock and predictions"""
    days_until_reorder = current_stock / np.mean(predicted_consumption) if np.mean(predicted_consumption) > 0 else float('inf')
    
    if current_stock <= reorder_level:
        return {
            'status': 'CRITICAL',
            'message': f'Cần đặt hàng ngay! Tồn kho ({current_stock:.0f}) đã dưới mức đặt hàng ({reorder_level:.0f})',
            'urgency': 'high'
        }
    elif current_stock <= reorder_level * (1 + safety_stock):
        return {
            'status': 'WARNING',
            'message': f'Nên đặt hàng sớm. Tồn kho ({current_stock:.0f}) gần đến mức đặt hàng ({reorder_level:.0f})',
            'urgency': 'medium'
        }
    else:
        return {
            'status': 'SAFE',
            'message': f'Tồn kho an toàn ({current_stock:.0f}). Còn {days_until_reorder:.1f} ngày trước khi cần đặt hàng',
            'urgency': 'low'
        }

def predict_consumption(model, data, reorder_levels, product_id, current_stock, seq_length=30):
    """Predict consumption with reorder level checks"""
    model.eval()
    with torch.no_grad():
        # Prepare input sequence
        input_seq = torch.FloatTensor(data[-seq_length:]).unsqueeze(0)
        
        # Get base prediction
        base_prediction = model(input_seq).squeeze().numpy()
        
        # Analyze patterns
        patterns = analyze_consumption_pattern(data)
        
        # Adjust prediction
        adjusted_prediction = adjust_prediction(base_prediction, patterns)
        
        # Get reorder level for the product
        reorder_level = reorder_levels.get(product_id, np.mean(data) * 0.2)  # Default to 20% of average if not found
        
        # Check reorder status
        reorder_status = check_reorder_status(current_stock, adjusted_prediction, reorder_level)
        
        return {
            'prediction': adjusted_prediction,
            'reorder_status': reorder_status,
            'daily_average': np.mean(adjusted_prediction),
            'confidence_interval': {
                'lower': np.percentile(adjusted_prediction, 25),
                'upper': np.percentile(adjusted_prediction, 75)
            }
        }

def calculate_metrics(actual, predicted):
    """Calculate prediction metrics"""
    mae = np.mean(np.abs(actual - predicted))
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape
    }

def save_model(model, path):
    """Save model"""
    torch.save(model.state_dict(), path)

def load_model(path, model):
    """Load model"""
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
