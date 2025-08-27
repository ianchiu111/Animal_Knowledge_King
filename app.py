import os
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
load_dotenv(".end")

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
@app.route('/api/v1/get_multiple_choice_question', methods=['GET'])
def get_questions():
    animal_type = request.args.get('animal_type')
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            if animal_type:
                cur.execute("SELECT * FROM multiple_choice_question WHERE animal_type = %s", (animal_type,))
            else:
                cur.execute("SELECT * FROM multiple_choice_question")
            questions = cur.fetchall()
            result = []
            for q in questions:
                cur2 = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                cur2.execute("SELECT player_name, comment, created_at FROM feedback WHERE question_id = %s", (q['id'],))
                feedbacks = [dict(row) for row in cur2.fetchall()]
                cur2.close()
                result.append({
                    "id": q['id'],
                    "animal_type": q['animal_type'],
                    "question": q['question'],
                    "options": [q['option1'], q['option2'], q['option3'], q['option4']],
                    "correct_answer": q['correct_answer'],
                    "explanation": q['explanation'],
                    "total_count": q['total_count'],
                    "count_option1": q['count_option1'],
                    "count_option2": q['count_option2'],
                    "count_option3": q['count_option3'],
                    "count_option4": q['count_option4'],
                    "feedbacks": feedbacks
                })
            return jsonify(result)

# ----------- 新增單選題 -------------
@app.route('/api/v1/add_new_multiple_choice_question', methods=['POST'])
def add_question():
    data = request.json
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO multiple_choice_question (animal_type, question, option1, option2, option3, option4,
                    correct_answer, explanation)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                RETURNING id
            """, (
                data['animal_type'], data['question'], data['option1'], data['option2'],
                data['option3'], data.get('option4'), data['correct_answer'], data['explanation']
            ))
            qid = cur.fetchone()[0]
            conn.commit()
    return jsonify({"message": "Question added", "id": qid})

# ----------- 新增回饋 -------------
@app.route('/api/v1/add_player_feedback', methods=['POST'])
def add_feedback():
    data = request.json
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO feedback (question_id, player_name, comment)
                VALUES (%s, %s, %s)
            """, (
                data['question_id'], data.get('player_name'), data['comment']
            ))
            conn.commit()
    return jsonify({"message": "Feedback added"})



# ========================== model ==========================
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
    return jsonify({"label": label})

# ========================== main ==========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
