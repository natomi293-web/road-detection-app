from flask import Flask, render_template, request, redirect, session
from roboflow import Roboflow
import supervision as sv
import cv2

app = Flask(__name__)
app.secret_key = "aqu126zhj923g"  # 適当な文字列でOK（セッション用）
PASSWORD = "DOURO12"  # ← 港さんが決めるパスワードに変更

from roboflow import Roboflow

rf = Roboflow(api_key="PNvjzdIq1rlRsoeBvXon")
project = rf.workspace("new-workspace-nep6p").project("one-lane-road-detecter-2")
model = project.version("rfdetr-small-t1").model

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
    if not session.get("logged_in"):   # ← これが「3」
        return redirect("/login")

    if request.method == "POST":
        file = request.files["image"]
        filepath = "static/upload.jpg"
        file.save(filepath)

        image = cv2.imread(filepath)
        results = model.infer(image)[0]
        detections = sv.Detections.from_inference(results)

        box_annotator = sv.BoxAnnotator(thickness=4)
        label_annotator = sv.LabelAnnotator(text_scale=1.5, text_thickness=2)

        labels = [p.class_name for p in results.predictions]

        annotated = box_annotator.annotate(scene=image, detections=detections)
        annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

        cv2.imwrite("static/result.jpg", annotated)

        return render_template("index.html", result=True)

    return render_template("index.html", result=False)

@app.route("/logout")
def logout():
    session.pop("logged_in", None)
    return redirect("/login")

if __name__ == "__main__":
    app.run(debug=True)


