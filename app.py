from flask import Flask, render_template, request, redirect, url_for, flash, has_request_context, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import json
import logging
import os
from forecasting import forecast_demand, generate_monthly_history
import hashlib
import uuid

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///inventory.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    role = db.Column(db.String(20), nullable=False)

class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    product_id = db.Column(db.String(20), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    category = db.Column(db.String(50), nullable=False)
    supplier_id = db.Column(db.String(20), nullable=False)
    supplier_name = db.Column(db.String(100), nullable=False)
    stock_quantity = db.Column(db.Integer, nullable=False)
    reorder_level = db.Column(db.Integer, nullable=False)
    reorder_quantity = db.Column(db.Integer, nullable=False)
    unit_price = db.Column(db.Float, nullable=False)
    date_received = db.Column(db.DateTime, nullable=False)
    last_order_date = db.Column(db.DateTime, nullable=False)
    expiration_date = db.Column(db.DateTime, nullable=False)
    warehouse_location = db.Column(db.String(100), nullable=False)
    sales_volume = db.Column(db.Integer, nullable=False)
    inventory_turnover_rate = db.Column(db.Float, nullable=False)
    status = db.Column(db.String(20), nullable=False)
    forecasted_demand = db.Column(db.Integer, nullable=True)
    last_forecast_date = db.Column(db.DateTime, nullable=True)
    historical_sales = db.Column(db.JSON, nullable=True)
    quality_metrics = db.Column(db.JSON, nullable=True)
    notes = db.Column(db.Text, nullable=True)
    suggested_location = db.Column(db.String(100), nullable=True)

class AuditLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    user = db.Column(db.String(80), nullable=False)
    action = db.Column(db.String(50), nullable=False)
    data = db.Column(db.Text, nullable=False)
    prev_hash = db.Column(db.String(64), nullable=True)
    hash = db.Column(db.String(64), nullable=False)
    note = db.Column(db.String(255), nullable=True)

    def __init__(self, user, action, data, prev_hash=None, note=None):
        self.user = user
        self.action = action
        self.data = data
        self.prev_hash = prev_hash
        self.note = note
        self.hash = self.compute_hash()

    def compute_hash(self):
        content = f"{self.timestamp}{self.user}{self.action}{self.data}{self.prev_hash or ''}{self.note or ''}"
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

class Batch(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    batch_id = db.Column(db.String(64), unique=True, nullable=False)
    product_id = db.Column(db.String(20), db.ForeignKey('product.product_id'), nullable=False)
    origin = db.Column(db.String(100), nullable=False)
    date_received = db.Column(db.DateTime, nullable=False)
    storage_condition = db.Column(db.String(100), nullable=True)
    note = db.Column(db.Text, nullable=True)
    # Relationship
    product = db.relationship('Product', backref='batches')

class BatchLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    batch_id = db.Column(db.String(64), db.ForeignKey('batch.batch_id'), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    action = db.Column(db.String(50), nullable=False)
    data = db.Column(db.Text, nullable=False)
    prev_hash = db.Column(db.String(64), nullable=True)
    hash = db.Column(db.String(64), nullable=False)
    note = db.Column(db.String(255), nullable=True)
    # Relationship
    batch = db.relationship('Batch', backref='logs')

    def __init__(self, batch_id, action, data, prev_hash=None, note=None):
        self.batch_id = batch_id
        self.action = action
        self.data = data
        self.prev_hash = prev_hash
        self.note = note
        self.hash = self.compute_hash()

    def compute_hash(self):
        content = f"{self.timestamp}{self.batch_id}{self.action}{self.data}{self.prev_hash or ''}{self.note or ''}"
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

# Routes
@app.route('/')
def index():
    # Chuyển hướng về trang xem tồn kho readonly
    return redirect(url_for('inventory_view'))

@app.route('/inventory_view')
def inventory_view():
    name = request.args.get('name', '').strip()
    product_id = request.args.get('product_id', '').strip()
    category = request.args.get('category', '').strip()
    status = request.args.get('status', '').strip()
    query = Product.query
    if name:
        query = query.filter(Product.name.ilike(f"%{name}%"))
    if product_id:
        query = query.filter(Product.product_id.ilike(f"%{product_id}%"))
    if category:
        query = query.filter(Product.category.ilike(f"%{category}%"))
    if status:
        query = query.filter(Product.status.ilike(f"%{status}%"))
    products = query.all()
    product_infos = []
    for p in products:
        info = analyze_inventory(p)
        product_infos.append({'product': p, **info})
    return render_template('inventory_readonly.html', product_infos=product_infos)

@app.route('/forecast')
def forecast():
    products = Product.query.all()
    warnings = []
    for product in products:
        # Lấy lịch sử bán hàng (nếu có), nếu không thì tạo từ sales_volume
        if product.historical_sales:
            sales_history = json.loads(product.historical_sales)
        else:
            sales_history = generate_monthly_history(product.sales_volume, months=12)
            product.historical_sales = json.dumps(sales_history)
        # Dự báo nhu cầu tháng tới
        forecasted = forecast_demand(sales_history, periods_ahead=1)
        product.forecasted_demand = forecasted
        product.last_forecast_date = datetime.now()
        # Cảnh báo thiếu/dư hàng
        if forecasted is not None:
            if forecasted > product.stock_quantity:
                warnings.append(f"{product.name} dự báo thiếu hàng: cần {forecasted}, tồn kho {product.stock_quantity}")
            elif forecasted < product.reorder_level:
                warnings.append(f"{product.name} dự báo dư thừa: dự báo {forecasted}, mức cảnh báo {product.reorder_level}")
    db.session.commit()
    # Dữ liệu cho biểu đồ
    forecast_data = pd.read_sql_query(
        """
        SELECT category, AVG(forecasted_demand) as avg_forecast, AVG(sales_volume) as avg_sales
        FROM product WHERE forecasted_demand IS NOT NULL GROUP BY category
        """,
        db.engine
    )
    forecast_chart = px.line(
        forecast_data,
        x='category',
        y=['avg_forecast', 'avg_sales'],
        title='Demand Forecast vs Actual Sales by Category',
        labels={'value': 'Quantity', 'variable': 'Metric'},
        color_discrete_map={'avg_forecast': 'blue', 'avg_sales': 'green'}
    ).to_json()
    return render_template('forecast.html', products=products, forecast_chart=forecast_chart, warnings=warnings)

@app.route('/traceability')
def traceability():
    return render_template('traceability.html')

@app.route('/delivery')
def delivery():
    return render_template('delivery.html')

@app.route('/contracts')
def contracts():
    return render_template('contracts.html')

@app.route('/reports')
def reports():
    # Get inventory statistics
    total_products = Product.query.count()
    total_value = db.session.query(db.func.sum(Product.stock_quantity * Product.unit_price)).scalar() or 0
    low_stock = Product.query.filter(Product.stock_quantity <= Product.reorder_level).count()
    expiring_soon = Product.query.filter(Product.expiration_date <= datetime.now()).count()
    
    # Get sales data by category
    sales_by_category = pd.read_sql_query(
        """
        SELECT category,
               SUM(sales_volume) as total_sales,
               AVG(inventory_turnover_rate) as avg_turnover
        FROM product
        GROUP BY category
        """,
        db.engine
    )
    
    # Create sales chart
    sales_chart = px.bar(
        sales_by_category,
        x='category',
        y='total_sales',
        title='Total Sales by Category',
        labels={'total_sales': 'Total Sales Volume'}
    ).to_json()
    
    # Create turnover chart
    turnover_chart = px.bar(
        sales_by_category,
        x='category',
        y='avg_turnover',
        title='Average Inventory Turnover by Category',
        labels={'avg_turnover': 'Turnover Rate'}
    ).to_json()
    
    # Get top selling products
    top_products = Product.query.order_by(Product.sales_volume.desc()).limit(10).all()
    
    # Get low stock products
    low_stock_products = Product.query.filter(
        Product.stock_quantity <= Product.reorder_level
    ).order_by(Product.stock_quantity.asc()).all()
    
    # Get expiring products
    expiring_products = Product.query.filter(
        Product.expiration_date <= datetime.now()
    ).order_by(Product.expiration_date.asc()).all()
    
    return render_template('reports.html',
                         total_products=total_products,
                         total_value=total_value,
                         low_stock=low_stock,
                         expiring_soon=expiring_soon,
                         sales_chart=sales_chart,
                         turnover_chart=turnover_chart,
                         top_products=top_products,
                         low_stock_products=low_stock_products,
                         expiring_products=expiring_products)

@app.route('/traceability/<batch_id>')
def traceability_batch(batch_id):
    batch = Batch.query.filter_by(batch_id=batch_id).first()
    if not batch:
        flash('Không tìm thấy batch này!', 'danger')
        return redirect(url_for('inventory_view'))
    logs = BatchLog.query.filter_by(batch_id=batch_id).order_by(BatchLog.timestamp.asc()).all()
    return render_template('traceability_batch.html', batch=batch, logs=logs)

def analyze_inventory(product):
    # Gợi ý vị trí sắp xếp dựa trên tần suất xuất kho
    if product.inventory_turnover_rate >= 10:
        product.suggested_location = 'Gần cửa/khu vực dễ lấy'
    elif product.inventory_turnover_rate >= 5:
        product.suggested_location = 'Kệ trung tâm'
    else:
        product.suggested_location = 'Kệ xa/kho phụ'
    # Gợi ý thời gian tái nhập hàng
    avg_daily_sale = product.sales_volume / 30 if product.sales_volume else 0.1
    days_left = product.stock_quantity / avg_daily_sale if avg_daily_sale else 9999
    restock_suggestion = None
    if days_left < 7:
        restock_suggestion = f'Nên tái nhập trong {int(days_left)} ngày tới'
    # Cảnh báo cận hạn sử dụng
    days_to_expire = (product.expiration_date - datetime.now()).days
    expire_warning = None
    if days_to_expire < 14:
        expire_warning = f'Sản phẩm sắp hết hạn sau {days_to_expire} ngày!'
    # Cảnh báo không luân chuyển
    days_since_last_order = (datetime.now() - product.last_order_date).days
    no_move_warning = None
    if days_since_last_order > 60:
        no_move_warning = f'Sản phẩm không xuất kho {days_since_last_order} ngày!'
    # Cảnh báo nguy cơ hư hỏng vật lý (giả lập: nếu có nhiều hoàn trả hoặc xuất hủy trong log)
    batch_ids = [b.batch_id for b in product.batches]
    recent_logs = []
    from sqlalchemy import or_
    if batch_ids:
        recent_logs = BatchLog.query.filter(
            BatchLog.batch_id.in_(batch_ids),
            or_(BatchLog.action == 'return', BatchLog.action == 'export'),
            BatchLog.timestamp > datetime.now() - timedelta(days=30)
        ).all()
    physical_warning = None
    if len([l for l in recent_logs if 'hỏng' in (l.note or '').lower() or 'lỗi' in (l.note or '').lower()]) > 2:
        physical_warning = 'CẢNH BÁO: Nhiều hoàn trả/hủy do hỏng/lỗi!'
    return {
        'suggested_location': product.suggested_location,
        'restock_suggestion': restock_suggestion,
        'expire_warning': expire_warning,
        'no_move_warning': no_move_warning,
        'physical_warning': physical_warning
    }

def init_db():
    with app.app_context():
        # Create database tables
        db.create_all()

if __name__ == '__main__':
    # Initialize database
    init_db()
    app.run(debug=True) 