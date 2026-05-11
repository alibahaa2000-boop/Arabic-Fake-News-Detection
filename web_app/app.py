
import os
import torch
from flask import Flask, request, render_template, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))

MODEL_DIR = r"D:\Desktop\Selenium\arabert_news_artifacts\model"

MAX_LENGTH = 256   
LABELS = {0: "كاذب", 1: "صحيح"}

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


@torch.no_grad()
def predict(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,   # ← fixed
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    logits = model(**inputs).logits
    probs  = torch.softmax(logits, dim=-1).squeeze(0)
    pred_id = int(torch.argmax(probs).item())
    conf    = float(probs[pred_id].item())
    return pred_id, conf, [float(probs[0].item()), float(probs[1].item())]


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    text = request.form.get("news", "").strip()

    if len(text) < 5:
        return render_template(
            "index.html",
            error="النص قصير جدًا. اكتب خبر أو فقرة أطول.",
            news=text,
        )

    pred_id, conf, probs = predict(text)
    result = LABELS[pred_id]

    return render_template(
        "index.html",
        news=text,
        result=result,
        confidence=f"{conf:.4f}",
        p_fake=f"{probs[0]:.4f}",
        p_true=f"{probs[1]:.4f}",
    )


@app.route("/predict_json", methods=["POST"])
def predict_json():
    data    = request.get_json(silent=True) or {}
    text    = str(data.get("text", "")).strip()

    if len(text) < 5:
        return jsonify({"ok": False, "error": "Text too short"}), 400

    pred_id, conf, probs = predict(text)
    return jsonify({
        "ok": True,
        "label_id": pred_id,
        "label": LABELS[pred_id],
        "confidence": conf,
        "p_fake": probs[0],
        "p_true": probs[1],
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)