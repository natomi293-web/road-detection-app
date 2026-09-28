from flask import Flask, render_template, request, redirect, session
import os
import cv2
from roboflow import Roboflow
import supervision as sv

app = Flask(__name__)
app.secret_key = "aqu126zhj923g"
PASSWORD = "DOURO12"

# Roboflow モデル読み込み
rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
project = rf.project("one-lane-road-detecter")
model = project.version(2)
# ===== モデル読み込みチェック =====
print("=== Roboflow Model Load Check ===")
print("API Key:", os.getenv("ROBOFLOW_API_KEY"))

try:
    print("Project:", project.name)
    print("Model loaded:", model is not None)
except Exception as e:
    print("Roboflow load error:", e)
# ==================================

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if request.form["password"] == PASSWORD:
            session["logged_in"] = True
            return redirect("/")
        else:
            return "パスワードが違います"
    return render_template("login.html")

@app.route("/", methods=["GET", "POST"])
def index():
    if not session.get("logged_in"):
        return redirect("/login")

    if request.method == "POST":
        file = request.files["image"]

        # static フォルダが無いと Render で落ちるため必ず作成
        os.makedirs("static", exist_ok=True)

        filepath = "static/upload.jpg"
        file.save(filepath)

        # 画像読み込み
        image = cv2.imread(filepath)
        if image is None:
            return "画像が読み込めませんでした（Render のパス問題）"

        # Roboflow 推論
        result = model.predict(filepath, hosted=True).json()

        if "predictions" not in result:
            return "Roboflow が予測を返しませんでした"

        predictions = result["predictions"]

        # supervision 用に変換
        detections = sv.Detections.from_inference(result["predictions"])

        # アノテーション作成
        box_annotator = sv.BoxAnnotator(thickness=4)
        label_annotator = sv.LabelAnnotator(text_scale=1.5, text_thickness=2)

        labels = [p["class"] for p in predictions]

        annotated = box_annotator.annotate(scene=image, detections=detections)
        annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

        output_path = os.path.join("static", "result.jpg")
        cv2.imwrite(output_path, annotated)

        return render_template("index.html", result=True)

    return render_template("index.html", result=False)
