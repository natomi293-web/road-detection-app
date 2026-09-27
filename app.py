from flask import Flask, render_template, request, redirect, session
from roboflow import Roboflow
import supervision as sv
import cv2
import os

app = Flask(__name__)
app.secret_key = "aqu126zhj923g"
PASSWORD = "DOURO12"

rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
project = rf.workspace("new-workspace-nep6p").project("one-lane-road-detecter")
model = project.version(2).model

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
        filepath = "static/upload.jpg"
        file.save(filepath)

        image = cv2.imread(filepath)

        result = model.predict(image).json()

        predictions = result["predictions"]
        
        detections = sv.Detections.from_inference(result["predictions"])

        box_annotator = sv.BoxAnnotator(thickness=4)
        label_annotator = sv.LabelAnnotator(text_scale=1.5, text_thickness=2)

        labels = [p["class"] for p in predictions]

        annotated = box_annotator.annotate(scene=image, detections=detections)
        annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

        cv2.imwrite("static/result.jpg", annotated)
        return render_template("index.html", result=True)

    return render_template("index.html", result=False)

