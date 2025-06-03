import pandas as pd
import qrcode
import os

df = pd.read_csv("Grocery_Inventory_and_Sales_Dataset.csv")
output_dir = "static/qr_codes"
os.makedirs(output_dir, exist_ok=True)

for product_id in df["Product_ID"]:
    product_id = str(product_id).strip()
    img = qrcode.make(product_id)
    img.save(f"{output_dir}/{product_id}.png")

print("✅ Đã tạo xong mã QR cho toàn bộ sản phẩm.")

