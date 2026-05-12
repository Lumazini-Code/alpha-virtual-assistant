import numpy as np
import onnxruntime as ortonnx_model
from tokenizers import Tokenizer
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# ── Carregar modelo ───────────────────────────────────────────────────────────

_tokenizer = Tokenizer.from_file("Models/DebertaV2ForSequenceClassification/tokenizer.json")
_tokenizer.enable_truncation()
_tokenizer.enable_padding()

_so = ort.SessionOptions()
_so.intra_op_num_threads = max(1, os.cpu_count() - 2)
_so.inter_op_num_threads = 1

_session = ort.InferenceSession(
    "Models/DebertaV2ForSequenceClassification/model.onnx",
    sess_options=_so,
    providers=["CPUExecutionProvider"]
)

_valid_inputs = {i.name for i in _session.get_inputs()}

# ── Inferência ────────────────────────────────────────────────────────────────

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def predict(text, labels):
    encodings = _tokenizer.encode_batch(
        [(text, label) for label in labels]
    )
    ort_inputs = {
        "input_ids":      np.array([e.ids for e in encodings], dtype=np.int64),
        "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
        "token_type_ids": np.array([e.type_ids for e in encodings], dtype=np.int64),
    }
    ort_inputs = {k: v for k, v in ort_inputs.items() if k in _valid_inputs}

    logits = _session.run(None, ort_inputs)[0]
    scores = softmax(logits[:, 0])

    results = [{"label": l, "score": float(s)} for l, s in zip(labels, scores)]
    results.sort(key=lambda x: x["score"], reverse=True)
    return results

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.route("/classify", methods=["POST"])
def classify():
    body = request.get_json()

    if not body:
        return jsonify({"error": "Body JSON ausente"}), 400

    text = body.get("text", "").strip()
    labels = body.get("labels", [])

    if not text:
        return jsonify({"error": "Campo 'text' ausente ou vazio"}), 400
    if not labels or not isinstance(labels, list):
        return jsonify({"error": "Campo 'labels' ausente ou inválido"}), 400

    results = predict(text, labels)

    return jsonify({
        "text": text,
        "best": results[0],
        "results": results
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3003)