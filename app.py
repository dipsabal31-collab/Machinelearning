import os, json, random
import numpy as np, pandas as pd
from flask import Flask, jsonify, render_template, request

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "app_data")
HF_IDS = {"MiniLM": "sentence-transformers/all-MiniLM-L6-v2", "MPNet": "sentence-transformers/all-mpnet-base-v2"}
app = Flask(__name__)

cfg = json.load(open(os.path.join(DATA, "config.json")))
LABELS = {int(k): v for k, v in cfg["label_names"].items()}
train = pd.read_csv(os.path.join(DATA, "train.csv")); test = pd.read_csv(os.path.join(DATA, "test.csv"))
ref_idx = np.load(os.path.join(DATA, "ood_ref_idx.npy"))

def load(path):
    x = np.load(path).astype("float32")
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

EMB = {m: {"train": load(f"{DATA}/{m}_train_emb.npy"), "test": load(f"{DATA}/{m}_test_emb.npy")}
       for m in cfg["ood_threshold"] if os.path.exists(f"{DATA}/{m}_train_emb.npy")}
ENC = {}

def query_vec(model, text, test_id):
    if test_id is not None:
        return EMB[model]["test"][int(test_id)]
    if model not in HF_IDS:
        raise ValueError(f"{model} cannot encode new text here (it needs a GPU). Pick a stored test article instead.")
    if model not in ENC:
        from sentence_transformers import SentenceTransformer
        ENC[model] = SentenceTransformer(HF_IDS[model], device="cpu")
    return ENC[model].encode([text], normalize_embeddings=True)[0].astype("float32")

def parse():
    d = request.get_json(force=True)
    model = d.get("model")
    if model not in EMB: raise ValueError("Unknown model.")
    text = (d.get("text") or "").strip()[:2000]; tid = d.get("test_id")
    if tid is None and not text: raise ValueError("Type some text or pick a stored article.")
    return model, query_vec(model, text, tid), d

@app.errorhandler(ValueError)
def bad(e): return jsonify(error=str(e)), 400

@app.route("/")
def index(): return render_template("index.html", models=list(EMB), live=[m for m in EMB if m in HF_IDS],
                                    results=json.load(open(os.path.join(BASE, "results.json"))))

@app.route("/api/random_test")
def random_test():
    i = random.randrange(len(test)); return jsonify(id=i, text=test.text[i], category=LABELS[int(test.label[i])])

@app.route("/api/search", methods=["POST"])
def search():
    model, q, d = parse(); k = max(1, min(int(d.get("k", 5)), 20))
    sims = EMB[model]["train"] @ q; top = np.argsort(-sims)[:k]
    cats = [LABELS[int(train.label[i])] for i in top]
    vote = max(set(cats), key=cats.count)
    return jsonify(predicted=vote, agreement=cats.count(vote) / k,
                   results=[{"rank": r + 1, "text": train.text[i], "category": cats[r], "similarity": float(sims[i])} for r, i in enumerate(top)])

@app.route("/api/ood", methods=["POST"])
def ood():
    model, q, d = parse()
    sims = EMB[model]["train"][ref_idx] @ q; j = int(np.argmax(sims)); thr = cfg["ood_threshold"][model]
    return jsonify(score=float(sims[j]), threshold=thr, in_domain=bool(sims[j] >= thr),
                   nearest=train.text[int(ref_idx[j])], nearest_category=LABELS[int(train.label[int(ref_idx[j])])])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
