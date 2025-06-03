import pandas as pd
from datetime import datetime
from app import app, db, Product, User, write_audit_log, Supplier
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
        logger.info(df.columns.tolist())
        
        # Convert date columns to datetime
        logger.info("Converting date columns...")
        date_columns = ['Date_Received', 'Last_Order_Date', 'Expiration_Date']
        for col in date_columns:
            df[col] = pd.to_datetime(df[col])
        
        # Clean up price column
        logger.info("Cleaning price data...")
        df['Unit_Price'] = df['Unit_Price'].str.replace('$', '').astype(float)
        
        # Fill null values in category
        df['Catagory'] = df['Catagory'].fillna('Unknown')
        
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
                # Check if supplier exists
                supplier = Supplier.query.filter_by(supplier_id=row['Supplier_ID']).first()
                if not supplier:
                    supplier = Supplier(
                        supplier_id=row['Supplier_ID'],
                        name=row['Supplier_Name'],
                        address='Unknown',  # Default value since not in CSV
                        phone='Unknown',    # Default value since not in CSV
                        email='Unknown'     # Default value since not in CSV
                    )
                    db.session.add(supplier)
                    db.session.flush()

                # Create product
                product = Product(
                    product_id=row['Product_ID'],
                    name=row['Product_Name'],
                    category=row['Catagory'],
                    supplier_id=row['Supplier_ID'],
                    stock_quantity=row['Stock_Quantity'],
                    reorder_level=row['Reorder_Level'],
                    reorder_quantity=row['Reorder_Quantity'],
                    unit_price=row['Unit_Price'],
                    date_received=row['Date_Received'],
                    last_order_date=row['Last_Order_Date'],
                    expiration_date=row['Expiration_Date'],
                    manufacturing_date=row['Date_Received'],  # Using date_received as manufacturing_date
                    warehouse_location=row['Warehouse_Location'],
                    sales_volume=row['Sales_Volume'],
                    inventory_turnover_rate=row['Inventory_Turnover_Rate'],
                    status=row['Status'],
                    note='Imported from CSV'
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
                    'category': row['Catagory'],
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
    with app.app_context():
        import_data() 