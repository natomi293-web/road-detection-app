from flask import Flask, render_template, request, redirect, session
import os
import cv2
import base64
from roboflow import Roboflow
import supervision as sv

app = Flask(__name__)
app.secret_key = "aqu126zhj923g"
PASSWORD = "DOURO12"

# Roboflow モデル読み込み
rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
project = rf.project("one-lane-road-detecter")
model = project.version(2)   # Cloud API は version オブジェクトで OK

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

        os.makedirs("static", exist_ok=True)
        filepath = "static/upload.jpg"
        file.save(filepath)

        # 画像読み込み
        image = cv2.imread(filepath)
        if image is None:
            return "画像が読み込めませんでした（Render のパス問題）"

        # ★ base64 に変換して Cloud API に渡す
        _, buffer = cv2.imencode(".jpg", image)
        base64_image = base64.b64encode(buffer).decode("utf-8")

        # ★ Cloud API 推論
        result = model.predict(base64_image, hosted=True).json()

        if "predictions" not in result:
            return "Roboflow が予測を返しませんでした"

        predictions = result["predictions"]

        # ★ supervision 用に predictions を手動で変換
        xyxy = []
        confidence = []
        class_id = []

        for p in predictions:
            x1 = p["x"] - p["width"] / 2
            y1 = p["y"] - p["height"] / 2
            x2 = p["x"] + p["width"] / 2
            y2 = p["y"] + p["height"] / 2

            xyxy.append([x1, y1, x2, y2])
            confidence.append(p.get("confidence", 1.0))
            class_id.append(0)

        detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id
        )

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

# ★ Render 用ポート設定（必須）
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
