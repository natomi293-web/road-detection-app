from flask import Flask, render_template, request, redirect, session
import os
import requests
from PIL import Image
import supervision as sv
import numpy as np
import io

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
        image_url = request.form.get("image_url")

        if not image_url:
            return "画像URLが受け取れていません"

        # Roboflow Cloud API (GET)
        url = f"https://detect.roboflow.com/{PROJECT_NAME}/{VERSION}"
        params = {
            "api_key": ROBOFLOW_API_KEY,
            "confidence": 40,
            "overlap": 30,
            "image": image_url
        }

        response = requests.get(url, params=params)
        result = response.json()

        print("=== Roboflow Response ===")
        print(result)

        if "predictions" not in result:
            return f"Roboflow が予測を返しませんでした: {result}"

        predictions = result["predictions"]

        # 画像をダウンロードして描画
        img_data = requests.get(image_url).content
        image = Image.open(io.BytesIO(img_data)).convert("RGB")
        np_image = np.array(image)

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

        box_annotator = sv.BoxAnnotator(thickness=4)
        label_annotator = sv.LabelAnnotator(text_scale=1.5, text_thickness=2)

        labels = [p["class"] for p in predictions]

        annotated = box_annotator.annotate(scene=np_image, detections=detections)
        annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

        output_path = os.path.join("static", "result.jpg")
        Image.fromarray(annotated).save(output_path)

        return render_template("index.html", result=True)

    return render_template("index.html", result=False)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
