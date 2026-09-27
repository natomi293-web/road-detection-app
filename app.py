from flask import Flask, render_template, request, redirect, session
from roboflow import Roboflow
import supervision as sv
import cv2
import os

app = Flask(__name__)
app.secret_key = "aqu126zhj923g"  # 適当な文字列でOK（セッション用）
PASSWORD = "DOURO12"  # ← 港さんが決めるパスワードに変更

# ★ Render の環境変数を使う（ハードコードしない）
rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))

# ★ workspace を指定しないと Cloud API は model=None になる
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

        # ★ infer() → predict() に変更（Cloud API は infer を持たない）
        result = model.predict(image).json()

        # ★ Cloud API の結果形式に合わせて変換
        predictions = result["predictions"]
        detections = sv.Detections.from_inference(result)

        box_annotator = sv.BoxAnnotator(thickness=4)
        label_annotator = sv.LabelAnnotator(text_scale=1.5, text_thickness=2)

        labels = [p["class"] for p in predictions]

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
    app.run(host="0.0.0.0", port=10000)
