from app import app, db, User
from werkzeug.security import generate_password_hash

def check_database():
    with app.app_context():
        # Check if admin user exists
        admin = User.query.filter_by(username='admin').first()
        if admin:
            print("Admin user found:")
            print(f"Username: {admin.username}")
            print(f"Role: {admin.role}")
            print(f"Password hash: {admin.password}")
        else:
            print("Admin user not found!")
            
            # Create admin user
            print("Creating admin user...")
            admin = User(
                username='admin',
                password=generate_password_hash('admin123'),
                role='admin'
            )
            db.session.add(admin)
            db.session.commit()
            print("Admin user created successfully!")

if __name__ == '__main__':
    check_database() 