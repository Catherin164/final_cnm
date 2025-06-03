import pandas as pd
from datetime import datetime
from app import app, db, Product, User, write_audit_log
from werkzeug.security import generate_password_hash
import logging


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def import_data():
    try:
        # Read the CSV file
        logger.info("Reading CSV file...")
        df = pd.read_csv('Grocery_Inventory_and_Sales_Dataset.csv')
        print(df.columns)
        
        # Convert date columns to datetime
        logger.info("Converting date columns...")
        df['Date_Received'] = pd.to_datetime(df['Date_Received'])
        df['Last_Order_Date'] = pd.to_datetime(df['Last_Order_Date'])
        df['Expiration_Date'] = pd.to_datetime(df['Expiration_Date'])
        
        # Clean up price column
        logger.info("Cleaning price data...")
        df['Unit_Price'] = df['Unit_Price'].str.replace('$', '').astype(float)
        
        with app.app_context():
            # Create admin user with hashed password
            logger.info("Creating admin user...")
            admin = User.query.filter_by(username='admin').first()
            if not admin:
                admin = User(
                    username='admin',
                    password=generate_password_hash('admin123'),
                    role='admin'
                )
                db.session.add(admin)
            
            # Import products
            logger.info("Importing products...")
            for _, row in df.iterrows():
                cat = row['Catagory'] if pd.notnull(row['Catagory']) else 'Unknown'
                product = Product(
                    product_id=row['Product_ID'],
                    name=row['Product_Name'],
                    category=cat,
                    supplier_id=row['Supplier_ID'],
                    supplier_name=row['Supplier_Name'],
                    stock_quantity=row['Stock_Quantity'],
                    reorder_level=row['Reorder_Level'],
                    reorder_quantity=row['Reorder_Quantity'],
                    unit_price=row['Unit_Price'],
                    date_received=row['Date_Received'],
                    last_order_date=row['Last_Order_Date'],
                    expiration_date=row['Expiration_Date'],
                    warehouse_location=row['Warehouse_Location'],
                    sales_volume=row['Sales_Volume'],
                    inventory_turnover_rate=row['Inventory_Turnover_Rate'],
                    status=row['Status'],
                    forecasted_demand=None,
                    last_forecast_date=None,
                    historical_sales=None,
                    quality_metrics=None,
                    notes=None
                )
                db.session.add(product)
                # Ghi log blockchain cho mỗi sản phẩm nhập kho
                try:
                    user = current_user.username if current_user.is_authenticated else 'system'
                except Exception:
                    user = 'system'
                log_data = {
                    'product_id': row['Product_ID'],
                    'name': row['Product_Name'],
                    'category': cat,
                    'quantity': row['Stock_Quantity'],
                    'date_received': str(row['Date_Received']),
                    'supplier': row['Supplier_Name']
                }
                write_audit_log('import', str(log_data), note='Import from CSV')
            
            # Commit all changes
            logger.info("Committing changes to database...")
            db.session.commit()
            logger.info("Data import completed successfully!")
            
    except Exception as e:
        logger.error(f"Error importing data: {str(e)}")
        raise

if __name__ == '__main__':
    import_data() 