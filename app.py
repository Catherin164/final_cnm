from flask import Flask, render_template, request, redirect, url_for, flash, has_request_context, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import json
from werkzeug.security import generate_password_hash, check_password_hash
import logging
import os
from forecasting import forecast_demand, generate_monthly_history, forecast_consumption, analyze_consumption_pattern, DemandForecaster
import hashlib
import uuid
import numpy as np
import joblib

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///inventory.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

##########
@app.route('/trace/<product_id>')
def trace_qr(product_id):
    df = pd.read_csv('Grocery_Inventory_and_Sales_Dataset.csv', dtype={'Product_ID': str})
    df['Product_ID'] = df['Product_ID'].astype(str).str.strip()
    clean_id = product_id.strip()

    product = df[df['Product_ID'] == clean_id]
    if product.empty:
        return render_template("not_found.html", product_id=clean_id)

########################

#qr scan
@app.route('/scan')
def scan():
    return render_template('scan_qr.html')


##########################

# Thêm filter enumerate
@app.template_filter('enumerate')
def enumerate_filter(iterable):
    return enumerate(iterable)

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    role = db.Column(db.String(20), nullable=False)

class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    product_id = db.Column(db.String(20), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    category = db.Column(db.String(50), nullable=False)
    supplier_id = db.Column(db.String(20), db.ForeignKey('supplier.supplier_id'), nullable=False)
    stock_quantity = db.Column(db.Integer, nullable=False)
    reorder_level = db.Column(db.Integer, nullable=True)
    reorder_quantity = db.Column(db.Integer, nullable=True)
    unit_price = db.Column(db.Float, nullable=False)
    date_received = db.Column(db.DateTime, nullable=False)
    last_order_date = db.Column(db.DateTime, nullable=True)
    expiration_date = db.Column(db.DateTime, nullable=False)
    manufacturing_date = db.Column(db.DateTime, nullable=False)
    warehouse_location = db.Column(db.String(100), nullable=False)
    sales_volume = db.Column(db.Integer, nullable=True)
    inventory_turnover_rate = db.Column(db.Float, nullable=True)
    status = db.Column(db.String(20), nullable=False)
    note = db.Column(db.Text, nullable=True)
    forecasted_demand = db.Column(db.Integer, nullable=True)
    last_forecast_date = db.Column(db.DateTime, nullable=True)
    historical_sales = db.Column(db.Text, nullable=True)
    # Relationship
    supplier = db.relationship('Supplier', backref='products')
    import_history = db.relationship('ImportHistory', backref='product')

class Supplier(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    supplier_id = db.Column(db.String(20), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    address = db.Column(db.String(200), nullable=False)
    phone = db.Column(db.String(20), nullable=False)
    email = db.Column(db.String(100), nullable=False)

class ImportHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    product_id = db.Column(db.String(20), db.ForeignKey('product.product_id'), nullable=False)
    batch_id = db.Column(db.String(64), nullable=False)
    quantity = db.Column(db.Integer, nullable=False)
    date_received = db.Column(db.DateTime, nullable=False)
    supplier_id = db.Column(db.String(20), db.ForeignKey('supplier.supplier_id'), nullable=False)
    note = db.Column(db.Text, nullable=True)
    # Relationship
    supplier = db.relationship('Supplier', backref='imports')

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

class Contract(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    contract_id = db.Column(db.String(20), unique=True, nullable=False)
    supplier_id = db.Column(db.String(20), db.ForeignKey('supplier.supplier_id'), nullable=False)
    sign_date = db.Column(db.DateTime, nullable=False)
    expiry_date = db.Column(db.DateTime, nullable=False)
    value = db.Column(db.Float, nullable=False)
    status = db.Column(db.String(20), nullable=False)  # pending, active, expired, completed
    terms = db.Column(db.Text, nullable=False)
    note = db.Column(db.Text, nullable=True)
    # Add bank details fields
    bank_name = db.Column(db.String(100), nullable=True)
    account_number = db.Column(db.String(50), nullable=True)
    # Smart contract fields
    recipient_wallet = db.Column(db.String(42), nullable=False)  # Ethereum wallet address
    delivery_deadline = db.Column(db.DateTime, nullable=False)
    payment_status = db.Column(db.String(20), nullable=False, default='pending')  # pending, paid, refunded
    delivery_status = db.Column(db.String(20), nullable=False, default='pending')  # pending, delivered, late
    smart_contract_address = db.Column(db.String(42), nullable=True)  # Deployed smart contract address
    # Relationship
    supplier = db.relationship('Supplier', backref='contracts')

class Wallet(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    address = db.Column(db.String(42), unique=True, nullable=False)
    balance = db.Column(db.Float, nullable=False, default=0.0)
    last_updated = db.Column(db.DateTime, nullable=False, default=datetime.now)

    def add_balance(self, amount):
        self.balance += amount
        self.last_updated = datetime.now()
        db.session.commit()

    def subtract_balance(self, amount):
        if self.balance >= amount:
            self.balance -= amount
            self.last_updated = datetime.now()
            db.session.commit()
            return True
        return False

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# Routes
@app.route('/')
@login_required
def index():
    # Get inventory statistics
    total_products = Product.query.count()
    low_stock = Product.query.filter(Product.stock_quantity <= Product.reorder_level).count()
    expiring_soon = Product.query.filter(Product.expiration_date <= datetime.now()).count()
    
    # Get sales data for charts
    sales_data = pd.read_sql_query(
        "SELECT category, SUM(sales_volume) as total_sales FROM product GROUP BY category",
        db.engine
    )
    
    sales_chart = px.bar(
        sales_data,
        x='category',
        y='total_sales',
        title='Sales by Category'
    ).to_json()
    
    return render_template('index.html',
                         total_products=total_products,
                         low_stock=low_stock,
                         expiring_soon=expiring_soon,
                         sales_chart=sales_chart)

@app.route('/inventory')
@login_required
def inventory():
    # Lấy tham số lọc từ request
    name = request.args.get('name', '').strip()
    product_id = request.args.get('product_id', '').strip()
    category = request.args.get('category', '').strip()
    status = request.args.get('status', '').strip()

    # Xây dựng truy vấn động
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
    return render_template('inventory.html', product_infos=product_infos)

@app.route('/forecast', methods=['GET', 'POST'])
def forecast():
    if request.method == 'GET':
        # Lấy danh sách sản phẩm cho form
        products = Product.query.all()
        
        # Lấy tham số từ request
        product_name = request.args.get('product')
        season = request.args.get('season')
        region = request.args.get('region')
        weather = request.args.get('weather')
        
        # Nếu không có tham số, hiển thị form
        if not all([product_name, season, region, weather]):
            return render_template('forecast.html', products=products)
            
        # Kiểm tra sản phẩm tồn tại
        product = Product.query.filter_by(name=product_name).first()
        if not product:
            return jsonify({'error': 'Sản phẩm không tồn tại'}), 404
            
        # Lấy dữ liệu lịch sử bán hàng
        historical_sales = []
        if product.historical_sales:
            try:
                historical_sales = json.loads(product.historical_sales)
            except:
                historical_sales = generate_monthly_history(product.sales_volume)
        else:
            historical_sales = generate_monthly_history(product.sales_volume)
            
        # Thực hiện dự báo
        forecaster = DemandForecaster()
        forecast_result = forecaster.forecast(historical_sales, periods_ahead=4)
        metrics = forecaster.get_forecast_metrics(historical_sales, forecast_result)
        pattern = analyze_consumption_pattern(historical_sales)
        
        # Tính toán số lượng cần nhập cho từng kỳ
        current_stock = product.stock_quantity
        import_quantities = []
        beginning_stocks = []
        ending_stocks = []
        pending_import = 0
        for idx, forecast in enumerate(forecast_result):
            # Nếu có nhập hàng từ kỳ trước, cộng vào đầu kỳ này
            if idx > 0:
                current_stock += pending_import
                pending_import = 0
            # Lưu tồn kho đầu kỳ
            beginning_stocks.append(current_stock)
            # Tính tồn kho cuối kỳ (chưa nhập hàng)
            ending_stock = max(0, current_stock - forecast)
            # Mặc định không nhập
            import_qty = 0
            # Nếu tồn kho cuối kỳ <= reorder_level, lên kế hoạch nhập cho đầu kỳ sau
            if ending_stock <= product.reorder_level:
                import_qty = max(product.reorder_quantity, forecast - current_stock)
                pending_import = import_qty
            import_quantities.append(import_qty)
            ending_stocks.append(ending_stock)
            current_stock = ending_stock
        # Cập nhật thông tin sản phẩm
        product.forecasted_demand = forecast_result[0] if forecast_result else 0
        product.last_forecast_date = datetime.now()
        product.historical_sales = json.dumps(historical_sales)
        db.session.commit()
        # Trả về kết quả
        return jsonify({
            'product': product.name,
            'current_stock': product.stock_quantity,
            'reorder_level': product.reorder_level,
            'last_forecast_date': product.last_forecast_date.strftime('%Y-%m-%d %H:%M:%S'),
            'forecast': {
                'best_model': forecaster.best_model,
                'metrics': metrics,
                'next_4_periods': forecast_result,
                'import_quantities': import_quantities,
                'beginning_stocks': beginning_stocks,
                'ending_stocks': ending_stocks
            },
            'pattern': pattern,
            'seasonality': pattern['seasonality']
        })
    return jsonify({'error': 'Method not allowed'}), 405

@app.route('/traceability')
@login_required
def traceability():
    product_id = request.args.get('product_id', '').strip()
    batch_id = request.args.get('batch_id', '').strip()

    product = None
    batch = None
    supplier = None
    batch_logs = []

    if product_id:
        product = Product.query.filter_by(product_id=product_id).first()
        if product:
            supplier = Supplier.query.filter_by(supplier_id=product.supplier_id).first()
    if batch_id:
        batch = Batch.query.filter_by(batch_id=batch_id).first()
        if batch:
            batch_logs = BatchLog.query.filter_by(batch_id=batch_id).order_by(BatchLog.timestamp.asc()).all()
            # Parse data field for each log
            import json
            for log in batch_logs:
                try:
                    log.data_parsed = json.loads(log.data) if log.data else {}
                except Exception:
                    log.data_parsed = {}

    return render_template(
        'traceability.html',
        product=product,
        batch=batch,
        supplier=supplier,
        batch_logs=batch_logs
    )

@app.route('/delivery')
@login_required
def delivery():
    return render_template('delivery.html')

@app.route('/reports')
@login_required
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

@app.route('/login', methods=['GET', 'POST'])
def login():
    logger.debug("Login route accessed")
    
    if current_user.is_authenticated:
        logger.debug("User already authenticated, redirecting to index")
        return redirect(url_for('index'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        logger.debug(f"Login attempt for username: {username}")
        
        if not username or not password:
            flash('Vui lòng nhập đầy đủ tên đăng nhập và mật khẩu', 'error')
            return render_template('login.html')
            
        user = User.query.filter_by(username=username).first()
        
        if user:
            logger.debug("User found in database")
            if check_password_hash(user.password, password):
                logger.debug("Password verified, logging in user")
                login_user(user)
                next_page = request.args.get('next')
                if not next_page or not next_page.startswith('/'):
                    next_page = url_for('index')
                return redirect(next_page)
            else:
                logger.debug("Password verification failed")
        else:
            logger.debug("User not found in database")
            
        flash('Tên đăng nhập hoặc mật khẩu không đúng', 'error')
            
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

def write_audit_log(action, data, note=None):
    last_log = AuditLog.query.order_by(AuditLog.id.desc()).first()
    prev_hash = last_log.hash if last_log else None
    if has_request_context() and hasattr(current_user, 'is_authenticated'):
        user = current_user.username if current_user.is_authenticated else 'system'
    else:
        user = 'system'
    log = AuditLog(user=user, action=action, data=data, prev_hash=prev_hash, note=note)
    db.session.add(log)
    db.session.commit()
    return log.hash

def init_db():
    with app.app_context():
        # Create database tables
        db.create_all()
        
        # Check if admin user exists
        admin = User.query.filter_by(username='admin').first()
        if not admin:
            logger.debug("Creating admin user...")
            admin = User(
                username='admin',
                password=generate_password_hash('admin123'),
                role='admin'
            )
            db.session.add(admin)
            db.session.commit()
            logger.debug("Admin user created successfully!")

@app.route('/import', methods=['POST'])
@login_required
def import_goods():
    product_id = request.form.get('product_id')
    quantity = int(request.form.get('quantity'))
    origin = request.form.get('origin', 'Unknown')
    storage_condition = request.form.get('storage_condition', '')
    note = request.form.get('note', '')
    product = Product.query.filter_by(product_id=product_id).first()
    if not product:
        flash('Sản phẩm không tồn tại', 'danger')
        return redirect(url_for('inventory'))
    product.stock_quantity += quantity
    db.session.commit()
    # Tạo batch mới
    batch_id = str(uuid.uuid4())
    batch = Batch(
        batch_id=batch_id,
        product_id=product_id,
        origin=origin,
        date_received=datetime.now(),
        storage_condition=storage_condition,
        note=note
    )
    db.session.add(batch)
    db.session.commit()
    # Ghi log truy xuất nguồn gốc
    log_data = {
        'product_id': product_id,
        'quantity': quantity,
        'origin': origin,
        'date_received': str(batch.date_received),
        'storage_condition': storage_condition,
        'note': note
    }
    write_batch_log(batch_id, 'import', str(log_data), note='Nhập kho tạo batch mới')
    write_audit_log('import', f'Import {quantity} to {product_id}', note='Nhập kho')
    flash(f'Đã nhập {quantity} sản phẩm vào kho (Batch: {batch_id})', 'success')
    return redirect(url_for('inventory'))

@app.route('/export', methods=['POST'])
@login_required
def export_goods():
    product_id = request.form.get('product_id')
    quantity = int(request.form.get('quantity'))
    export_date = datetime.strptime(request.form.get('export_date'), '%Y-%m-%d')
    receiving_unit = request.form.get('receiving_unit')
    reason = request.form.get('reason')
    product = Product.query.filter_by(product_id=product_id).first()
    if not product:
        flash('Sản phẩm không tồn tại', 'danger')
        return redirect(url_for('inventory'))
    abnormal = False
    note = f'Xuất kho ngày {export_date.strftime("%d/%m/%Y")} cho {receiving_unit}, lý do: {reason}'
    if quantity > product.stock_quantity:
        abnormal = True
        note += ' | CẢNH BÁO: Xuất vượt tồn kho!'
    product.stock_quantity -= quantity
    db.session.commit()
    write_audit_log('export', f'Export {quantity} from {product_id}', note=note)
    if abnormal:
        flash('CẢNH BÁO: Xuất vượt tồn kho!', 'danger')
    else:
        flash(f'Đã xuất {quantity} sản phẩm khỏi kho cho {receiving_unit}', 'success')
    return redirect(url_for('inventory'))

@app.route('/edit_product', methods=['POST'])
@login_required
def edit_product():
    product_id = request.form.get('product_id')
    new_quantity = int(request.form.get('new_quantity'))
    product = Product.query.filter_by(product_id=product_id).first()
    if not product:
        flash('Sản phẩm không tồn tại', 'danger')
        return redirect(url_for('inventory'))
    abnormal = False
    note = 'Chỉnh sửa số lượng'
    if abs(new_quantity - product.stock_quantity) > 100:  # Rule: chỉnh sửa lớn hơn 100 là bất thường
        abnormal = True
        note += ' | CẢNH BÁO: Chỉnh sửa số lượng lớn bất thường!'
    product.stock_quantity = new_quantity
    db.session.commit()
    write_audit_log('edit', f'Edit quantity of {product_id} to {new_quantity}', note=note)
    if abnormal:
        flash('CẢNH BÁO: Chỉnh sửa số lượng lớn bất thường!', 'danger')
    else:
        flash('Đã cập nhật số lượng sản phẩm', 'success')
    return redirect(url_for('inventory'))

@app.route('/return', methods=['POST'])
@login_required
def return_goods():
    product_id = request.form.get('product_id')
    quantity = int(request.form.get('quantity'))
    return_unit = request.form.get('return_unit')
    return_date = datetime.strptime(request.form.get('return_date'), '%Y-%m-%d')
    reason = request.form.get('reason')
    
    product = Product.query.filter_by(product_id=product_id).first()
    if not product:
        flash('Sản phẩm không tồn tại', 'danger')
        return redirect(url_for('inventory'))
    
    product.stock_quantity += quantity
    db.session.commit()
    
    note = f'Hoàn trả hàng từ {return_unit}, ngày {return_date.strftime("%d/%m/%Y")}, lý do: {reason}'
    write_audit_log('return', f'Return {quantity} to {product_id}', note=note)
    
    flash(f'Đã hoàn trả {quantity} sản phẩm vào kho. Lý do: {reason}', 'info')
    return redirect(url_for('inventory'))

def analyze_inventory(product):
    # Gợi ý vị trí sắp xếp dựa trên tần suất xuất kho
    turnover_rate = product.inventory_turnover_rate or 0  # Default to 0 if None
    if turnover_rate >= 10:
        product.suggested_location = 'Gần cửa/khu vực dễ lấy'
    elif turnover_rate >= 5:
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
    days_since_last_order = (datetime.now() - product.last_order_date).days if product.last_order_date else 0
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

def write_batch_log(batch_id, action, data, note=None):
    last_log = BatchLog.query.filter_by(batch_id=batch_id).order_by(BatchLog.id.desc()).first()
    prev_hash = last_log.hash if last_log else None
    log = BatchLog(batch_id=batch_id, action=action, data=data, prev_hash=prev_hash, note=note)
    db.session.add(log)
    db.session.commit()
    return log.hash

@app.route('/traceability/<batch_id>')
@login_required
def traceability_batch(batch_id):
    batch = Batch.query.filter_by(batch_id=batch_id).first()
    if not batch:
        flash('Không tìm thấy batch này!', 'danger')
        return redirect(url_for('inventory'))
    logs = BatchLog.query.filter_by(batch_id=batch_id).order_by(BatchLog.timestamp.asc()).all()
    return render_template('traceability_batch.html', batch=batch, logs=logs)

@app.route('/add_new_product', methods=['POST'])
def add_new_product():
    try:
        # Lấy thông tin từ form
        product_id = request.form.get('product_id')
        batch_id = request.form.get('batch_id')
        name = request.form.get('name')
        category = request.form.get('category')
        manufacturing_date = datetime.strptime(request.form.get('manufacturing_date'), '%Y-%m-%d')
        expiration_date = datetime.strptime(request.form.get('expiration_date'), '%Y-%m-%d')
        quantity = int(request.form.get('quantity'))
        unit_price = float(request.form.get('unit_price'))
        note = request.form.get('note')

        # Thông tin nhà cung cấp
        supplier_id = request.form.get('supplier_id')
        supplier_name = request.form.get('supplier_name')
        supplier_address = request.form.get('supplier_address')
        supplier_phone = request.form.get('supplier_phone')
        supplier_email = request.form.get('supplier_email')

        # Kiểm tra xem sản phẩm đã tồn tại chưa
        existing_product = Product.query.filter_by(product_id=product_id).first()
        if existing_product:
            flash('Mã sản phẩm đã tồn tại!', 'error')
            return redirect(url_for('inventory'))

        # Kiểm tra và thêm nhà cung cấp mới nếu chưa tồn tại
        supplier = Supplier.query.filter_by(supplier_id=supplier_id).first()
        if not supplier:
            supplier = Supplier(
                supplier_id=supplier_id,
                name=supplier_name,
                address=supplier_address,
                phone=supplier_phone,
                email=supplier_email
            )
            db.session.add(supplier)
            db.session.flush()  # Để lấy ID của supplier mới

        # Tính toán các giá trị mặc định
        current_date = datetime.now()
        reorder_level = max(quantity // 4, 1)  # 25% số lượng nhập
        reorder_quantity = max(quantity // 2, 1)  # 50% số lượng nhập
        warehouse_location = f"Khu vực {category}"  # Vị trí mặc định theo danh mục

        # Tạo dự báo ban đầu cho sản phẩm mới
        initial_sales_history = generate_monthly_history(quantity, months=12)
        forecast_result = forecast_demand(initial_sales_history, periods_ahead=1)
        if isinstance(forecast_result, dict) and 'forecast' in forecast_result:
            forecasted = forecast_result['forecast'][0] if forecast_result['forecast'] else quantity
        else:
            forecasted = forecast_result if isinstance(forecast_result, (int, float)) else quantity

        # Tạo sản phẩm mới
        new_product = Product(
            product_id=product_id,
            name=name,
            category=category,
            stock_quantity=quantity,
            reorder_level=reorder_level,
            reorder_quantity=reorder_quantity,
            unit_price=unit_price,
            date_received=current_date,
            last_order_date=current_date,
            expiration_date=expiration_date,
            manufacturing_date=manufacturing_date,
            warehouse_location=warehouse_location,
            sales_volume=quantity,
            inventory_turnover_rate=0.0,  # Bắt đầu từ 0
            status='Còn hàng',
            supplier_id=supplier_id,
            note=note,
            forecasted_demand=forecasted,
            last_forecast_date=current_date,
            historical_sales=json.dumps(initial_sales_history)
        )

        # Tạo lịch sử nhập hàng
        import_history = ImportHistory(
            product_id=product_id,
            batch_id=batch_id,
            quantity=quantity,
            date_received=current_date,
            supplier_id=supplier_id,
            note=note
        )

        # Lưu vào database
        db.session.add(new_product)
        db.session.add(import_history)
        db.session.commit()

        flash('Thêm sản phẩm mới thành công!', 'success')
        return redirect(url_for('inventory'))

    except Exception as e:
        db.session.rollback()
        flash(f'Có lỗi xảy ra: {str(e)}', 'error')
        return redirect(url_for('inventory'))

@app.route('/audit_log')
@login_required
def audit_log():
    # Lấy tham số lọc từ request
    user = request.args.get('user', '').strip()
    action = request.args.get('action', '').strip()
    start_date = request.args.get('start_date', '').strip()
    end_date = request.args.get('end_date', '').strip()
    page = request.args.get('page', 1, type=int)
    per_page = 20  # Số bản ghi mỗi trang

    # Xây dựng truy vấn
    query = AuditLog.query

    # Áp dụng các bộ lọc
    if user:
        query = query.filter(AuditLog.user.ilike(f"%{user}%"))
    if action:
        query = query.filter(AuditLog.action == action)
    if start_date:
        query = query.filter(AuditLog.timestamp >= datetime.strptime(start_date, '%Y-%m-%d'))
    if end_date:
        query = query.filter(AuditLog.timestamp <= datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1))

    # Sắp xếp theo thời gian mới nhất
    query = query.order_by(AuditLog.timestamp.desc())

    # Phân trang
    total = query.count()
    pages = (total + per_page - 1) // per_page
    logs = query.paginate(page=page, per_page=per_page).items

    return render_template('audit_log.html',
                         logs=logs,
                         pages=pages,
                         current_page=page)

@app.route('/smart_contracts')
@login_required
def smart_contracts():
    contracts = Contract.query.filter(Contract.recipient_wallet.isnot(None)).all()
    suppliers = Supplier.query.all()
    
    # Get wallet information
    wallets = {w.address: w for w in Wallet.query.all()}
    
    return render_template('smart_contracts.html', 
                         contracts=contracts, 
                         suppliers=suppliers,
                         wallets=wallets)

@app.route('/create_smart_contract', methods=['POST'])
@login_required
def create_smart_contract():
    contract_id = request.form.get('contract_id')

    # Kiểm tra hợp đồng đã tồn tại
    existing_contract = Contract.query.filter_by(contract_id=contract_id).first()
    if existing_contract:
        flash(f'Mã hợp đồng {contract_id} đã tồn tại! Vui lòng chọn mã khác.', 'error')
        return redirect(url_for('smart_contracts'))

    try:
        supplier_id = request.form.get('supplier_id')
        account_number = request.form.get('account_number') # Lấy số tài khoản
        bank_name = request.form.get('bank_name') # Lấy tên ngân hàng
        value = float(request.form.get('value'))
        delivery_deadline = datetime.strptime(request.form.get('delivery_deadline'), '%Y-%m-%d')
        terms = request.form.get('terms')

        # Tạo địa chỉ ví từ username của người dùng hiện tại
        recipient_wallet = f"0x{hashlib.sha256(current_user.username.encode()).hexdigest()[:40]}"

        # Create or get wallet
        wallet = Wallet.query.filter_by(address=recipient_wallet).first()
        if not wallet:
            wallet = Wallet(address=recipient_wallet, balance=0.0)
            db.session.add(wallet)
            db.session.flush()

        # Create new contract
        new_contract = Contract(
            contract_id=contract_id,
            supplier_id=supplier_id,
            sign_date=datetime.now(),
            expiry_date=delivery_deadline,
            value=value,
            status='pending',
            terms=terms,
            recipient_wallet=recipient_wallet,
            delivery_deadline=delivery_deadline,
            account_number=account_number, # Gán số tài khoản
            bank_name=bank_name # Gán tên ngân hàng
        )

        # Simulate smart contract deployment
        new_contract.smart_contract_address = f"0x{hashlib.sha256(contract_id.encode()).hexdigest()[:40]}"

        db.session.add(new_contract)
        db.session.commit()

        write_audit_log('smart_contract', f'Created smart contract {contract_id}',
                       note=f'Contract value: {value:,.0f} VNĐ, Deadline: {delivery_deadline.strftime("%d/%m/%Y")}')

        flash('Tạo hợp đồng thông minh thành công!', 'success')
        return redirect(url_for('smart_contracts'))

    except Exception as e:
        db.session.rollback()
        flash(f'Có lỗi xảy ra: {str(e)}', 'error')
        return redirect(url_for('smart_contracts'))

@app.route('/update_contract_status/<contract_id>', methods=['POST'])
@login_required
def update_contract_status(contract_id):
    try:
        contract = Contract.query.filter_by(contract_id=contract_id).first()
        if not contract:
            flash('Không tìm thấy hợp đồng!', 'error')
            return redirect(url_for('smart_contracts'))

        delivery_status = request.form.get('delivery_status')
        if delivery_status in ['delivered', 'late']:
            contract.delivery_status = delivery_status
            
            # Get recipient wallet
            wallet = Wallet.query.filter_by(address=contract.recipient_wallet).first()
            if not wallet:
                wallet = Wallet(address=contract.recipient_wallet, balance=0.0)
                db.session.add(wallet)
                db.session.flush()

            # Xử lý thanh toán tự động
            if delivery_status == 'delivered' and datetime.now() <= contract.delivery_deadline:
                # Kiểm tra số dư ví
                if wallet.balance >= contract.value:
                    # Trừ tiền từ ví
                    wallet.subtract_balance(contract.value)
                    contract.payment_status = 'paid'
                    contract.status = 'completed'
                    
                    # Ghi log thanh toán
                    write_audit_log('payment', 
                                  f'Automatic payment for contract {contract_id}', 
                                  note=f'Amount: {contract.value:,.0f} VNĐ, New balance: {wallet.balance:,.0f} VNĐ')
                    
                    flash('Giao hàng đúng hạn! Thanh toán tự động đã được thực hiện.', 'success')
                else:
                    flash('Số dư ví không đủ để thanh toán!', 'warning')
                    contract.payment_status = 'pending'
            elif delivery_status == 'late':
                contract.payment_status = 'refunded'
                contract.status = 'expired'
                flash('Giao hàng trễ hạn! Hợp đồng đã bị hủy.', 'warning')

            db.session.commit()
            write_audit_log('contract_status', 
                          f'Updated contract {contract_id} status', 
                          note=f'Delivery: {delivery_status}, Payment: {contract.payment_status}')

        return redirect(url_for('smart_contracts'))

    except Exception as e:
        db.session.rollback()
        flash(f'Có lỗi xảy ra: {str(e)}', 'error')
        return redirect(url_for('smart_contracts'))

@app.route('/add_balance', methods=['POST'])
@login_required
def add_balance():
    try:
        wallet_address = request.form.get('wallet_address')
        amount = float(request.form.get('amount'))
        
        # Get or create wallet
        wallet = Wallet.query.filter_by(address=wallet_address).first()
        if not wallet:
            wallet = Wallet(address=wallet_address, balance=0.0)
            db.session.add(wallet)
            db.session.flush()
        
        # Add balance
        wallet.add_balance(amount)
        
        write_audit_log('wallet', f'Added {amount:,.0f} VNĐ to wallet {wallet_address[:10]}...{wallet_address[-8:]}', 
                       note=f'New balance: {wallet.balance:,.0f} VNĐ')
        
        flash(f'Đã nạp {amount:,.0f} VNĐ vào ví. Số dư mới: {wallet.balance:,.0f} VNĐ', 'success')
        return redirect(url_for('smart_contracts'))
        
    except Exception as e:
        db.session.rollback()
        flash(f'Có lỗi xảy ra: {str(e)}', 'error')
        return redirect(url_for('smart_contracts'))

@app.route('/create_purchase_order', methods=['POST'])
@login_required
def create_purchase_order():
    try:
        product_id = request.form.get('product_id')
        supplier_id = request.form.get('supplier_id')
        quantity = int(request.form.get('quantity'))
        expected_delivery_date = datetime.strptime(request.form.get('expected_delivery_date'), '%Y-%m-%d')
        note = request.form.get('note', '')

        # Kiểm tra sản phẩm và nhà cung cấp
        product = Product.query.filter_by(product_id=product_id).first()
        supplier = Supplier.query.filter_by(supplier_id=supplier_id).first()
        
        if not product or not supplier:
            flash('Không tìm thấy thông tin sản phẩm hoặc nhà cung cấp!', 'error')
            return redirect(url_for('forecast'))

        # Tạo mã đơn hàng
        order_id = f"PO{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Tạo hợp đồng thông minh
        contract = Contract(
            contract_id=order_id,
            supplier_id=supplier_id,
            sign_date=datetime.now(),
            expiry_date=expected_delivery_date,
            value=quantity * product.unit_price,
            status='pending',
            terms=f"Đặt hàng {quantity} {product.name}",
            note=note,
            recipient_wallet=current_user.username,  # Sử dụng username làm địa chỉ ví
            delivery_deadline=expected_delivery_date
        )

        db.session.add(contract)
        db.session.commit()

        # Ghi log
        write_audit_log('purchase_order', 
                       f'Created purchase order {order_id}', 
                       note=f'Product: {product.name}, Quantity: {quantity}, Expected delivery: {expected_delivery_date.strftime("%d/%m/%Y")}')

        flash(f'Đã tạo đơn đặt hàng {order_id} thành công!', 'success')
        return redirect(url_for('smart_contracts'))

    except Exception as e:
        db.session.rollback()
        flash(f'Có lỗi xảy ra: {str(e)}', 'error')
        return redirect(url_for('forecast'))

@app.route('/dubaonhucau', methods=['GET', 'POST'])
def dubaonhucau():
    df = pd.read_csv('Grocery_Inventory_and_Sales_Dataset.csv')
    products = sorted(df['Product'].unique())
    seasons = ['Xuân', 'Hạ', 'Thu', 'Đông']
    areas = sorted(df['Area'].unique())
    weathers = ['Nắng', 'Mưa', 'Âm u', 'Lạnh']

    result_table = None
    selected = {'product': '', 'season': '', 'area': '', 'weather': ''}
    num_weeks = 2  # Số tuần muốn dự báo giống ảnh

    if request.method == 'POST':
        selected['product'] = request.form.get('product')
        selected['season'] = request.form.get('season')
        selected['area'] = request.form.get('area')
        selected['weather'] = request.form.get('weather')

        # Lọc dữ liệu theo lựa chọn
        filtered = df[
            (df['Product'] == selected['product']) &
            (df['Area'] == selected['area'])
        ]
        tonkho = 100  # Tồn kho hiện tại (giả lập)
        result_table = []
        for i in range(num_weeks):
            # Lấy trung bình 4 dòng gần nhất (hoặc logic khác tùy bạn)
            weekly_forecast = int(filtered['Quantity'].tail(4).mean()) if not filtered.empty else 0
            need_import = max(weekly_forecast - tonkho, 0)
            status = 'Sắp hết' if need_import > 0 else 'Đủ'
            result_table.append({
                'week': f'Tuần {i+1}',
                'forecast': weekly_forecast,
                'stock': tonkho,
                'need_import': need_import,
                'status': status
            })
            # Cập nhật tồn kho cho tuần tiếp theo
            tonkho = max(tonkho - weekly_forecast + need_import, 0)

    return render_template('dubaonhucau.html',
                           products=products,
                           seasons=seasons,
                           areas=areas,
                           weathers=weathers,
                           selected=selected,
                           result_table=result_table)

if __name__ == '__main__':
    # Initialize database
    init_db()
    app.run(debug=True)

