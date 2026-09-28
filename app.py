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
project = rf.workspace("new-workspace-nep6p").project("one-lane-road-detecter")
model = project.version(2).model

# ログインページ
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if request.form["password"] == PASSWORD:
            session["logged_in"] = True
            return redirect("/")
        else:
            return "パスワードが違います"
    return render_template("login.html")

# メインページ
@app.route("/", methods=["GET", "POST"])
def index():
    if not session.get("logged_in"):
        return redirect("/login")

    if request.method == "POST":
        file = request.files["image"]
        filepath = "static/upload.jpg"
        file.save(filepath)

        image = cv2.imread(filepath)
        result = model.predict(image).json()

        if "predictions" not in result:
            return "Roboflow が予測を返しませんでした"

        predictions = result["predictions"]

        # Detections 作成（インデント修正済み）
        detections = sv.Detections.from_inference(predictions)

        box_annotator = sv.BoxAnnotator(thickness=4)
        label_annotator = sv.LabelAnnotator(text_scale=1.5, text_thickness=2)

        labels = [p["class"] for p in predictions]

        annotated = box_annotator.annotate(scene=image, detections=detections)
        annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

        output_path = os.path.join(os.getcwd(), "static", "result.jpg")
        cv2.imwrite(output_path, annotated)

        return render_template("index.html", result=True)

    return render_template("index.html", result=False)
