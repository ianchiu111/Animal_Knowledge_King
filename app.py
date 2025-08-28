import os
import requests
from transformers import AutoModelForImageClassification, AutoImageProcessor
import torch
from PIL import Image
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timezone
import psycopg2
import psycopg2.extras

from dotenv import load_dotenv
load_dotenv(".env")

#  Postgres Params
DB_HOST = os.getenv("DB_HOST")
DB_DATABASE = os.getenv("DB_DATABASE")
DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT")

def get_conn():
    return psycopg2.connect(
        host=DB_HOST,
        database=DB_DATABASE,
        user=DB_USERNAME,
        password=DB_PASSWORD,
        port=DB_PORT
    )

# 初始化模型
model = AutoModelForImageClassification.from_pretrained("backend/src/model/vit-finetuned").eval()
image_processor = AutoImageProcessor.from_pretrained("backend/src/model/vit-finetuned")

app = Flask(__name__)
CORS(app)  ## 讓前端能跨網域存取 API


# ========================== Postgres ==========================

# ----------- 取得單選題 -------------
## 根據 label (model 辨識完回覆的動物種類) 取得隨機一題
## ⚠️ 目前只有設計取得單選題
## ⚠️ 不要用 f-string 或 + 拼接 SQL 參數的做法！ ➡️ 標準、安全的做法

@app.route('/api/v1/get_multiple_choice_question', methods=['GET'])
def get_questions():
    animal_type = request.args.get('animal_type')

    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            try:
                cur.execute(
                    """
                    SELECT id, animal_type, question, option1, option2, option3, option4, correct_answer, explanation
                    FROM multiple_choice_question
                    WHERE animal_type = %s
                    ORDER BY RANDOM()
                    LIMIT 1
                    """,
                    (animal_type,)
                )
            except Exception as e:
                print("Error occurred while fetching question:", e)
                return jsonify({"label": animal_type, "question": None})


            q = cur.fetchone()
            if q is None:
                print(f"No questions found for animal type: {animal_type} and {q}")
                return jsonify({"animal_type": animal_type, "question": None})

            # options list：option4 可能為 None
            question_data = {
                "id": q['id'],
                "animal_type": q['animal_type'],
                "question": q['question'],
                "options": [q['option1'], q['option2'], q['option3'], q['option4']],
                "correct_answer": str(q['correct_answer']).strip(),
                "explanation": q['explanation'],
            }

            print("API - get_multiple_choice_question by animal type", question_data)

            return jsonify({"animal_type": animal_type, "question": question_data})


# # ----------- 新增單選題 -------------
# @app.route('/api/v1/add_new_multiple_choice_question', methods=['POST'])
# def add_question():
#     data = request.json
#     with get_conn() as conn:
#         with conn.cursor() as cur:
#             cur.execute("""
#                 INSERT INTO multiple_choice_question (animal_type, question, option1, option2, option3, option4,
#                     correct_answer, explanation)
#                 VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
#                 RETURNING id
#             """, (
#                 data['animal_type'], data['question'], data['option1'], data['option2'],
#                 data['option3'], data.get('option4'), data['correct_answer'], data['explanation']
#             ))
#             qid = cur.fetchone()[0]
#             conn.commit()
#     return jsonify({"message": "Question added", "id": qid})

# # ----------- 新增回饋 -------------
# @app.route('/api/v1/add_player_feedback', methods=['POST'])
# def add_feedback():
#     data = request.json
#     with get_conn() as conn:
#         with conn.cursor() as cur:
#             cur.execute("""
#                 INSERT INTO feedback (question_id, player_name, comment)
#                 VALUES (%s, %s, %s)
#             """, (
#                 data['question_id'], data.get('player_name'), data['comment']
#             ))
#             conn.commit()
#     return jsonify({"message": "Feedback added"})



# ========================== model ==========================
## 前端拍照完進入的 model 辨識 endpoint
@app.route("/api/v1/predict_image_category", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img = Image.open(file.stream).convert("RGB")
    inputs = image_processor(img, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    pred = logits.argmax(-1).item()
    label = model.config.id2label[pred]

    print("辨識出來的 label:", label)

    # 從本機呼叫 API
    try:
        resp = requests.get(
            "http://localhost:5000/api/v1/get_multiple_choice_question",
            params={"animal_type": label}
        )
        data = resp.json()
    except Exception as e:
        print("Error occurred while calling get_multiple_choice_question:", e)
        return jsonify({"label": label, "question": None})

    # 回傳資料格式統一
    return jsonify({"label": label, "question": data.get("question")})

# ========================== main ==========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
