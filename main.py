from inference import get_model
import supervision as sv
import cv2

# ① 自分のモデルを読み込む
model = get_model(
    model_id="new-workspace-nep6p/one-lane-road-detecter-2-rfdetr-small-t1",  # ← Roboflowの画面に出ているIDに合わせて書き換え
    api_key="PNvjzdIq1rlRsoeBvXon"               # ← RoboflowのAPIキーを入れる
)

# ② 画像を読み込む（同じフォルダに test.jpg を置いておく）
image = cv2.imread("test.jpg")

# ③ 推論を実行
results = model.infer(image)[0]

# ④ Supervision用に変換
detections = sv.Detections.from_inference(results)

# ⑤ アノテータを作成
box_annotator = sv.BoxAnnotator(
    thickness=4,          # 枠線を太くする
    color=sv.Color.red()  # 色を赤にする（見やすい）
)

label_annotator = sv.LabelAnnotator(
    text_scale=1.5,       # 文字サイズを大きく
    text_thickness=2,     # 文字の太さ
    text_color=sv.Color.white(),  # 文字色
    background_color=sv.Color.black()  # 背景を黒にして視認性UP
)


# ⑥ ラベルを取り出す
labels = [p.class_name for p in results.predictions]

# ⑦ 画像にバウンディングボックスとラベルを描画
annotated_image = box_annotator.annotate(
    scene=image,
    detections=detections
)
annotated_image = label_annotator.annotate(
    scene=annotated_image,
    detections=detections,
    labels=labels
)

# ⑧ 結果を表示
sv.plot_image(annotated_image)
