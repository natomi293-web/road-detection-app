from flask import Flask, render_template, request, redirect, session
import os
import base64
import requests
from PIL import Image
import io
import supervision as sv
import numpy as np

app = Flask(__name__)
app.secret_key = "aqu126zhj923g"
PASSWORD = "DOURO12"

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
PROJECT_NAME = "one-lane-road-detecter"
VERSION = 2

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

        # ★ 画像をバイトとして直接読み込む（Render で壊れない）
        file_bytes = file.read()
        if len(file_bytes) == 0:
            return "画像が壊れています（0バイト）"

        try:
            image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        except:
            return "画像が読み込めませんでした（Pillow 読み込み失敗）"

        # ★ Pillow → JPEG → base64
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # ★ Roboflow Cloud API (POST)
        url = f"https://detect.roboflow.com/{PROJECT_NAME}/{VERSION}"
        params = {
            "api_key": ROBOFLOW_API_KEY,
            "confidence": 40,
            "overlap": 30
        }
        data = {
            "image": base64_image
        }

        response = requests.post(url, params=params, json=data)
        result = response.json()

        print("=== Roboflow Response ===")
        print(result)

        if "predictions" not in result:
            return f"Roboflow が予測を返しませんでした: {result}"

        predictions = result["predictions"]

        # supervision 用に変換
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

        # Pillow → numpy
        np_image = np.array(image)

        box_annotator = sv.BoxAnnotator(thickness=4)
        label_annotator = sv.LabelAnnotator(text_scale=1.5, text_thickness=2)

        labels = [p["class"] for p in predictions]

        annotated = box_annotator.annotate(scene=np_image, detections=detections)
        annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

        # ★ 結果画像を保存
        output_path = os.path.join("static", "result.jpg")
        Image.fromarray(annotated).save(output_path)

        return render_template("index.html", result=True)

    return render_template("index.html", result=False)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
