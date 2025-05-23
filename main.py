from sqlalchemy import create_engine, exc
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from supabase import create_client, Client
from flask import Flask, jsonify, request

app = Flask(__name__)

# Supabase setup
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_ANON_KEY")
supabase: Client = create_client(url, key)

# ✅ Sử dụng connection pooler (cổng 6543)
connection_string = "postgresql://postgres.alzndhssbtncurvytlqe:d7ITajxfCNIu2Rod@aws-0-us-east-2.pooler.supabase.com:6543/postgres"

try:
    # Kết nối với timeout 10 giây
    engine = create_engine(connection_string, connect_args={"connect_timeout": 10})
    conn = engine.connect()

    query = "SELECT * FROM product"
    df_sanpham = pd.read_sql(query, conn)

    print(df_sanpham.head())

except exc.SQLAlchemyError as e:
    print("Lỗi khi kết nối hoặc truy vấn cơ sở dữ liệu:", e)

finally:
    if 'conn' in locals():
        conn.close()
        print("Đã đóng kết nối.")

feature = ['price', 'description']

def combine_features(row):
    return str(row['price']) + ' ' + str(row['description'])

df_sanpham['combined_features'] = df_sanpham.apply(combine_features, axis=1)

print(df_sanpham['combined_features'].head())

tf = TfidfVectorizer()
tfMatrix = tf.fit_transform(df_sanpham['combined_features'])

similar = cosine_similarity(tfMatrix)
number = 5

@app.route('/api', methods=['GET'])
def get_data():
    ket_qua = []

    product_id = request.args.get('id')  # ✅ sửa lại để lấy param từ query string
    if not product_id:
        return jsonify({'error': 'Thiếu tham số id'})

    try:
        product_id = int(product_id)
    except ValueError:
        return jsonify({'error': 'id phải là số nguyên'})

    if product_id not in df_sanpham['id'].values:
        return jsonify({'error': 'id không hợp lệ'})

    indexproduct = df_sanpham[df_sanpham['id'] == product_id].index[0]
    similarProduct = list(enumerate(similar[indexproduct]))

    sorted_similarProduct = sorted(similarProduct, key=lambda x: x[1], reverse=True)

    def lay_ten(index):
        return df_sanpham.loc[index]['title']

    for i in range(1, number + 1):
        ket_qua.append(lay_ten(sorted_similarProduct[i][0]))

    data = {'san pham goi y': ket_qua}
    return jsonify(data)

if __name__ == '__main__':
    # ✅ Cho phép truy cập từ thiết bị khác trong mạng LAN
    app.run(host="0.0.0.0", port=5555)
