import os
import socket
import pandas as pd
from flask import Flask, jsonify, request
from sqlalchemy import create_engine, exc
from dotenv import set_key
from supabase import create_client, Client
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD

app = Flask(__name__)

# Supabase setup
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_ANON_KEY")
supabase: Client = create_client(url, key)

# PostgreSQL connection via Supabase Pooler
connection_string = "postgresql://postgres.alzndhssbtncurvytlqe:d7ITajxfCNIu2Rod@aws-0-us-east-2.pooler.supabase.com:6543/postgres"

try:
    engine = create_engine(connection_string, connect_args={"connect_timeout": 10})
    conn = engine.connect()

    df_sanpham = pd.read_sql("SELECT * FROM product", conn)
    df_danhgia = pd.read_sql("SELECT user, product, rating FROM product_review WHERE rating IS NOT NULL", conn)

    print("✅ Đã tải dữ liệu sản phẩm và đánh giá.")

except exc.SQLAlchemyError as e:
    print("❌ Lỗi kết nối DB:", e)
    df_sanpham = pd.DataFrame()
    df_danhgia = pd.DataFrame()

finally:
    if 'conn' in locals():
        conn.close()

# ==== Content-Based Recommendation ====
feature_cols = ['price', 'description', 'category']

def combine_features(row):
    return f"{row['price']} {row['description']} {row['category']}"

df_sanpham['combined_features'] = df_sanpham.apply(combine_features, axis=1)
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df_sanpham['combined_features'])
cosine_sim = cosine_similarity(tfidf_matrix)

@app.route('/api/content-based', methods=['GET'])
def content_based_api():
    product_id = request.args.get('id')
    if not product_id:
        return jsonify({'error': 'Thiếu tham số id'})

    try:
        product_id = int(product_id)
    except ValueError:
        return jsonify({'error': 'id phải là số nguyên'})

    if product_id not in df_sanpham['id'].values:
        return jsonify({'error': 'id không hợp lệ'})

    index = df_sanpham[df_sanpham['id'] == product_id].index[0]
    similarity_scores = list(enumerate(cosine_sim[index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    top_similar_titles = [
        df_sanpham.iloc[i[0]]['title'] for i in similarity_scores[1:6]
    ]

    return jsonify({'goi_y_noi_dung': top_similar_titles})


# ==== Collaborative Filtering Recommendation ====
def get_collaborative_recommendations(user_id, top_n=5):
    # Convert user_id to string
    user_id = str(user_id)
    
    # Convert user column to string
    df_danhgia['user'] = df_danhgia['user'].astype(str)
    
    print(f"Debug - User ID received: {user_id}")
    print(f"Debug - Available users: {df_danhgia['user'].unique()}")
    
    if df_danhgia.empty:
        print("Debug - No ratings data available")
        return []

    if user_id not in df_danhgia['user'].values:
        print(f"Debug - User {user_id} not found in ratings data")
        return []

    try:
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(df_danhgia[['user', 'product', 'rating']], reader)
        trainset = data.build_full_trainset()

        algo = SVD()
        algo.fit(trainset)

        all_product_ids = df_danhgia['product'].unique()
        rated_products = df_danhgia[df_danhgia['user'] == user_id]['product'].values
        unrated_products = [pid for pid in all_product_ids if pid not in rated_products]

        print(f"Debug - User {user_id} has rated {len(rated_products)} products")
        print(f"Debug - Found {len(unrated_products)} unrated products")

        predictions = [algo.predict(user_id, pid) for pid in unrated_products]
        filtered_predictions = [pred for pred in predictions if pred.est >= 4.0]
        filtered_predictions.sort(key=lambda x: x.est, reverse=True)

        recommended_product_ids = [pred.iid for pred in filtered_predictions[:top_n]]
        titles = df_sanpham[df_sanpham['id'].isin(recommended_product_ids)]['title'].tolist()
        
        print(f"Debug - Generated {len(titles)} recommendations")
        return titles

    except Exception as e:
        print(f"Debug - Error in collaborative filtering: {str(e)}")
        return []

@app.route('/api/collaborative', methods=['GET'])
def collaborative_api():
    user_id = request.args.get('user')
    if not user_id:
        return jsonify({'error': 'Thiếu tham số user'}), 400

    print(f"Debug - API called with user_id: {user_id}")
    result = get_collaborative_recommendations(user_id)
    
    if not result:
        return jsonify({
            'error': 'Không tìm thấy gợi ý cho người dùng này',
            'user_id': user_id
        }), 404

    return jsonify({
        'goi_y_cong_tac': result,
        'user_id': user_id
    })


# ==== IP Ghi vào .env ====
def get_lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def update_env_file(ip):
    env_path = ".env"
    if not os.path.exists(env_path):
        with open(env_path, "w") as f:
            f.write("")
    set_key(env_path, "LOCAL_IPV4", ip)

if __name__ == '__main__':
    lan_ip = get_lan_ip()
    update_env_file(lan_ip)
    print(f"✅ Địa chỉ IP LAN: {lan_ip} (đã ghi vào .env)")
    app.run(host="0.0.0.0", port=5555)
