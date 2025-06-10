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
    print(f"📦 Số sản phẩm: {len(df_sanpham)} | Số đánh giá: {len(df_danhgia)}")

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
    print(f"📩 Nhận request content-based với id: {product_id}")

    if not product_id:
        print("❌ Thiếu tham số id")
        return jsonify({'error': 'Thiếu tham số id'}), 400

    try:
        product_id = int(product_id)
    except ValueError:
        print("❌ id phải là số nguyên")
        return jsonify({'error': 'id phải là số nguyên'}), 400

    print(f"🔎 Danh sách id sản phẩm hợp lệ: {df_sanpham['id'].tolist()}")
    if product_id not in df_sanpham['id'].values:
        print(f"❌ id không hợp lệ: {product_id}")
        return jsonify({'error': 'id không hợp lệ'}), 400

    index = df_sanpham[df_sanpham['id'] == product_id].index[0]
    similarity_scores = list(enumerate(cosine_sim[index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    top_similar_titles = [
        df_sanpham.iloc[i[0]]['title'] for i in similarity_scores[1:6]
    ]

    print(f"🔍 Gợi ý dựa trên nội dung cho ID {product_id}: {top_similar_titles}")
    return jsonify({'goi_y_noi_dung': top_similar_titles})

# ==== Collaborative Filtering Recommendation ====
# ==== Gợi ý collaborative bằng SVD ====
def get_collaborative_recommendations(user_id, top_n=5):
    try:
        user_id = str(user_id)

        # Tải lại dữ liệu đánh giá
        with engine.connect() as conn:
            df_danhgia = pd.read_sql("SELECT user, product, rating FROM product_review WHERE rating IS NOT NULL", conn)

        # Ép kiểu user về string để khớp với user_id
        df_danhgia['user'] = df_danhgia['user'].astype(str)

        print(f"📩 Nhận request collaborative với user_id: {user_id}")
        print(f"🔍 Tổng số người dùng: {df_danhgia['user'].nunique()}")
        print(f"🔎 Các user có đánh giá: {df_danhgia['user'].unique().tolist()}")

        if df_danhgia.empty:
            print("⚠️ Không có dữ liệu đánh giá")
            return []

        # Setup Surprise Dataset
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(df_danhgia[['user', 'product', 'rating']], reader)
        trainset = data.build_full_trainset()

        # Kiểm tra user_id có tồn tại trong tập huấn luyện
        inner_user_ids = [trainset.to_raw_uid(uid) for uid in trainset.all_users()]
        if user_id not in inner_user_ids:
            print(f"⚠️ Người dùng {user_id} không tồn tại trong tập huấn luyện")
            return []

        # Huấn luyện mô hình
        algo = SVD()
        algo.fit(trainset)
        print("✅ Mô hình SVD đã huấn luyện")

        # Xác định sản phẩm chưa được đánh giá
        all_product_ids = df_sanpham['id'].unique()
        rated_products = df_danhgia[df_danhgia['user'] == user_id]['product'].values
        unrated_products = [pid for pid in all_product_ids if pid not in rated_products]

        print(f"🔍 Người dùng {user_id} đã đánh giá {len(rated_products)} sản phẩm")
        print(f"🆕 Có {len(unrated_products)} sản phẩm chưa đánh giá")

        if not unrated_products:
            print("⚠️ Không có sản phẩm chưa đánh giá để gợi ý")
            return []

        # Dự đoán và lọc theo threshold
        predictions = [algo.predict(user_id, pid) for pid in unrated_products]
        filtered_predictions = [pred for pred in predictions if pred.est >= 4.0]
        filtered_predictions.sort(key=lambda x: x.est, reverse=True)

        recommended_product_ids = [int(pred.iid) for pred in filtered_predictions[:top_n]]

        print(f"📊 Điểm dự đoán >= 4.0: {[round(p.est, 2) for p in filtered_predictions]}")
        print(f"🎯 ID sản phẩm gợi ý: {recommended_product_ids}")

        matched_products = df_sanpham[df_sanpham['id'].isin(recommended_product_ids)]
        titles = matched_products['title'].tolist()

        print(f"🎁 Gợi ý tiêu đề sản phẩm: {titles}")
        return titles

    except Exception as e:
        print(f"❌ Lỗi khi xử lý collaborative filtering: {str(e)}")
        return []

# ==== Gợi ý từ sản phẩm yêu thích cá nhân ====
def get_favorite_recommendations(user_id, top_n=5):
    try:
        with engine.connect() as conn:
            df_fav = pd.read_sql("""
                SELECT product 
                FROM favorite_product 
                WHERE user = %s
                ORDER BY created_at DESC
                LIMIT %s
            """, conn, params=(user_id, top_n))

        if df_fav.empty:
            print(f"⚠️ User {user_id} chưa yêu thích sản phẩm nào.")
            return []

        matched = df_sanpham[df_sanpham['id'].isin(df_fav['product'])]
        titles = matched['title'].tolist()
        print(f"❤️ Gợi ý từ sản phẩm yêu thích của {user_id}: {titles}")
        return titles

    except Exception as e:
        print(f"❌ Lỗi khi xử lý favorite recommendations: {str(e)}")
        return []

# ==== Gợi ý phổ biến dựa trên lượt yêu thích nhiều nhất ====
def get_trending_products(top_n=5):
    try:
        with engine.connect() as conn:
            df_trending = pd.read_sql("""
                SELECT product, COUNT(*) AS total 
                FROM favorite_product 
                GROUP BY product 
                ORDER BY total DESC 
                LIMIT %s
            """, conn, params=(top_n,))
        
        if df_trending.empty:
            print("⚠️ Không có dữ liệu sản phẩm phổ biến.")
            return []

        matched = df_sanpham[df_sanpham['id'].isin(df_trending['product'])]
        titles = matched['title'].tolist()
        print(f"🔥 Gợi ý sản phẩm phổ biến: {titles}")
        return titles

    except Exception as e:
        print(f"❌ Lỗi khi xử lý trending recommendations: {str(e)}")
        return []


# ==== Flask API ====
@app.route('/api/collaborative', methods=['GET'])
def collaborative_api():
    user_id = request.args.get('user')
    print(f"📥 API gọi với user: {user_id}")

    if not user_id:
        print("❌ Thiếu tham số user")
        return jsonify({'error': 'Thiếu tham số user'}), 400

    result_collab = get_collaborative_recommendations(user_id)
    result_fav = get_favorite_recommendations(user_id)
    result_trending = get_trending_products()

    print(f"✅ Trả về {len(result_collab)} gợi ý collaborative, "
          f"{len(result_fav)} yêu thích, {len(result_trending)} phổ biến.")

    return jsonify({
        'user_id': user_id,
        'goi_y_cong_tac': result_collab,
        'goi_y_yeu_thich': result_fav,
        'goi_y_pho_bien': result_trending
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
    app.run(host="0.0.0.0", port=5555, debug=True)
