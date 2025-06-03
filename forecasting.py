import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings('ignore')

class DemandForecaster:
    def __init__(self):
        self.models = {
            'linear': LinearRegression(),
            'arima': None,
            'exponential': None
        }
        self.best_model = None
        self.best_score = float('inf')
        
    def prepare_data(self, sales_history):
        """Chuẩn bị dữ liệu cho việc dự báo"""
        if not isinstance(sales_history, pd.Series):
            sales_history = pd.Series(sales_history)
        return sales_history
    
    def train_models(self, sales_history):
        """Huấn luyện các mô hình dự báo"""
        data = self.prepare_data(sales_history)
        
        # Linear Regression với đa thức bậc 2
        X = np.arange(len(data)).reshape(-1, 1)
        X_poly = np.hstack([X, X**2])  # Thêm bậc 2
        y = data.values
        self.models['linear'].fit(X_poly, y)
        
        # ARIMA với tham số tự động
        try:
            # Thử các tham số khác nhau cho ARIMA
            best_aic = float('inf')
            best_order = None
            for p in range(2):
                for d in range(2):
                    for q in range(2):
                        try:
                            model = ARIMA(data, order=(p,d,q))
                            results = model.fit()
                            if results.aic < best_aic:
                                best_aic = results.aic
                                best_order = (p,d,q)
                        except:
                            continue
            
            if best_order:
                self.models['arima'] = ARIMA(data, order=best_order)
                self.models['arima'] = self.models['arima'].fit()
        except:
            self.models['arima'] = None
            
        # Exponential Smoothing với tham số tối ưu
        try:
            best_aic = float('inf')
            best_params = None
            for trend in ['add', 'mul']:
                for seasonal in ['add', 'mul']:
                    try:
                        model = ExponentialSmoothing(
                            data,
                            seasonal_periods=12,
                            trend=trend,
                            seasonal=seasonal
                        ).fit()
                        if model.aic < best_aic:
                            best_aic = model.aic
                            best_params = (trend, seasonal)
                    except:
                        continue
            
            if best_params:
                self.models['exponential'] = ExponentialSmoothing(
                    data,
                    seasonal_periods=12,
                    trend=best_params[0],
                    seasonal=best_params[1]
                ).fit()
        except:
            self.models['exponential'] = None
            
        
        self._evaluate_models(data)
        
    def _evaluate_models(self, data):
        """Đánh giá các mô hình và chọn mô hình tốt nhất"""
        train_size = int(len(data) * 0.8)
        train_data = data[:train_size]
        test_data = data[train_size:]
        
        for model_name, model in self.models.items():
            if model is None:
                continue
                
            try:
                if model_name == 'linear':
                    X_test = np.arange(train_size, len(data)).reshape(-1, 1)
                    X_test_poly = np.hstack([X_test, X_test**2])
                    predictions = model.predict(X_test_poly)
                elif model_name == 'arima':
                    predictions = model.forecast(steps=len(test_data))
                else:  # exponential
                    predictions = model.forecast(len(test_data))
                
                # Tính toán nhiều metrics
                mae = mean_absolute_error(test_data, predictions)
                rmse = np.sqrt(mean_squared_error(test_data, predictions))
                r2 = r2_score(test_data, predictions)
                
                # Tính điểm tổng hợp
                score = mae * 0.4 + rmse * 0.4 + (1 - r2) * 0.2
                
                if score < self.best_score:
                    self.best_score = score
                    self.best_model = model_name
            except:
                continue
    
    def forecast(self, sales_history, periods_ahead=1):
        """Thực hiện dự báo sử dụng mô hình tốt nhất"""
        if isinstance(sales_history, pd.Series):
            if sales_history.empty or len(sales_history) < 3:
                return None
        else:
            if not sales_history or len(sales_history) < 3:
                return None
                
        data = self.prepare_data(sales_history)
        self.train_models(data)
        
        if self.best_model == 'linear':
            X = np.arange(len(data), len(data) + periods_ahead).reshape(-1, 1)
            X_poly = np.hstack([X, X**2])
            forecast = self.models['linear'].predict(X_poly)
        elif self.best_model == 'arima':
            forecast = self.models['arima'].forecast(steps=periods_ahead)
        else:  # exponential
            forecast = self.models['exponential'].forecast(periods_ahead)
            
        
        forecast = [max(0, round(x)) for x in forecast]
        
        
        noise = np.random.normal(0, 0.1, periods_ahead)
        forecast = [max(0, round(x * (1 + n))) for x, n in zip(forecast, noise)]
        
        return forecast
    
    def get_forecast_metrics(self, sales_history, forecast):
        """Tính toán các metrics đánh giá dự báo"""
        if not sales_history or not forecast:
            return None
            
        actual = sales_history[-len(forecast):]
        mae = mean_absolute_error(actual, forecast)
        rmse = np.sqrt(mean_squared_error(actual, forecast))
        r2 = r2_score(actual, forecast)
        
        return {
            'mae': round(mae, 2),
            'rmse': round(rmse, 2),
            'r2': round(r2, 2),
            'accuracy': round((1 - mae/np.mean(actual)) * 100, 2) if np.mean(actual) > 0 else 0
        }
    
    def analyze_seasonality(self, sales_history):
        """Phân tích tính mùa vụ của dữ liệu"""
        if isinstance(sales_history, pd.Series):
            if sales_history.empty or len(sales_history) < 12:
                return None
        else:
            if not sales_history or len(sales_history) < 12:
                return None
            
        data = self.prepare_data(sales_history)
        
        # Tính toán các chỉ số mùa vụ
        # Giả sử dữ liệu được sắp xếp theo thứ tự tháng từ 1-12
        monthly_means = []
        for i in range(12):
            if i < len(data):
                monthly_means.append(data.iloc[i] if isinstance(data, pd.Series) else data[i])
            else:
                monthly_means.append(0)
                
        monthly_means = pd.Series(monthly_means)
        mean_value = monthly_means.mean()
        seasonal_index = monthly_means / mean_value if mean_value > 0 else pd.Series([1] * 12)
        
        # Xác định mùa cao điểm và thấp điểm
        peak_month = seasonal_index.idxmax() + 1  # +1 vì index bắt đầu từ 0
        low_month = seasonal_index.idxmin() + 1
        
        return {
            'seasonal_strength': round((seasonal_index.max() - seasonal_index.min()) * 100, 2),
            'peak_month': int(peak_month),
            'low_month': int(low_month),
            'seasonal_pattern': 'Mạnh' if (seasonal_index.max() - seasonal_index.min()) > 0.3 else 'Yếu'
        }

def forecast_demand(sales_history, periods_ahead=1):
    """Hàm wrapper cho việc dự báo nhu cầu"""
    if isinstance(sales_history, pd.Series):
        if sales_history.empty or len(sales_history) < 3:
            return None
    else:
        if not sales_history or len(sales_history) < 3:
            return None
            
    forecaster = DemandForecaster()
    forecast = forecaster.forecast(sales_history, periods_ahead)
    metrics = forecaster.get_forecast_metrics(sales_history, forecast)
    seasonality = forecaster.analyze_seasonality(sales_history)
    
    return {
        'forecast': forecast,
        'metrics': metrics,
        'seasonality': seasonality,
        'best_model': forecaster.best_model
    }

def analyze_consumption_pattern(sales_history):
    """Phân tích mẫu tiêu thụ"""
    if isinstance(sales_history, pd.Series):
        if sales_history.empty or len(sales_history) < 3:
            return None
    else:
        if not sales_history or len(sales_history) < 3:
            return None
    
    data = pd.Series(sales_history)
    
    # Tính các chỉ số cơ bản
    total_sales = data.sum()
    avg_monthly = data.mean()
    std_dev = data.std()
    
    # Phân tích xu hướng
    X = np.array(range(len(data))).reshape(-1, 1)
    y = data.values
    model = LinearRegression()
    model.fit(X, y)
    trend = model.coef_[0]
    
    # Xác định mẫu tiêu thụ
    if trend > 0.1 * avg_monthly:
        pattern = "Tăng"
    elif trend < -0.1 * avg_monthly:
        pattern = "Giảm"
    else:
        pattern = "Ổn định"
    
    # Tính độ biến động
    volatility = std_dev / avg_monthly if avg_monthly > 0 else 0
    
    # Phân tích mùa vụ
    forecaster = DemandForecaster()
    seasonality = forecaster.analyze_seasonality(data)
    
    return {
        'pattern': pattern,
        'volatility': round(volatility * 100, 1),
        'trend_strength': round(abs(trend / avg_monthly * 100), 1) if avg_monthly > 0 else 0,
        'avg_monthly': round(avg_monthly, 1),
        'std_dev': round(std_dev, 1),
        'seasonality': seasonality
    }


def generate_monthly_history(sales_volume, months=12):
   
    base = sales_volume / 12 
    seasonal_factors = [1.2, 1.1, 1.0, 0.9, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.2, 1.1]
    noise = np.random.normal(0, 0.1, months)
    history = [base * seasonal_factors[i] * (1 + noise[i]) for i in range(months)]
    return [max(0, round(x)) for x in history]  

def forecast_consumption(sales_history, periods_ahead=4):
    
    if not sales_history or len(sales_history) < 3:
        return None
    
    # Tính toán các chỉ số
    total_sales = sum(sales_history)
    avg_monthly = total_sales / len(sales_history)
    avg_weekly = avg_monthly / 4
    
    # Phân tích xu hướng
    X = np.array(range(len(sales_history))).reshape(-1, 1)
    y = np.array(sales_history)
    model = LinearRegression()
    model.fit(X, y)
    
    # Dự báo cho tuần tiếp theo
    future_X = np.array(range(len(sales_history), len(sales_history) + 1)).reshape(-1, 1)
    weekly_forecast = model.predict(future_X)[0] / 4  # Chia cho 4 để có dự báo tuần
    
    # Tính toán các chỉ số bổ sung
    trend = model.coef_[0]  # Hệ số xu hướng
    seasonality = np.std(sales_history) / np.mean(sales_history) if np.mean(sales_history) > 0 else 0
    
    # Tạo dự báo chi tiết
    forecast_details = {
        'weekly_forecast': max(0, round(weekly_forecast)),  # Dự báo tuần tiếp theo
        'avg_weekly': round(avg_weekly, 1),
        'trend': round(trend, 2),
        'seasonality': round(seasonality, 2),
        'confidence': calculate_confidence(sales_history, [weekly_forecast])
    }
    
    return forecast_details

def calculate_confidence(sales_history, forecast):
    """Tính toán độ tin cậy của dự báo"""
    if not sales_history or len(sales_history) < 3:
        return 0
    
    # Tính độ lệch chuẩn của lịch sử
    std_dev = np.std(sales_history)
    mean_sales = np.mean(sales_history)
    
    # Tính hệ số biến thiên
    cv = std_dev / mean_sales if mean_sales > 0 else 1
    
    # Tính độ tin cậy (1 - cv, giới hạn trong khoảng 0-1)
    confidence = max(0, min(1, 1 - cv))
    
    return round(confidence * 100)  # Chuyển thành phần trăm 