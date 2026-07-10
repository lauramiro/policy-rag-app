import os

from flask import Flask, jsonify, render_template, request, send_from_directory

from rag.chain import answer_question, get_llm, make_retrieve_fn
from rag.config import Config
from rag.ingest import load_or_build_vectorstore

app = Flask(__name__)

_resources: dict = {}


def get_resources() -> dict:
    if "config" not in _resources:
        config = Config()
        vectorstore = load_or_build_vectorstore(config)
        _resources["config"] = config
        _resources["retrieve_fn"] = make_retrieve_fn(vectorstore, config)
        _resources["llm"] = get_llm(config)
    return _resources


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/corpus/<path:filename>")
def corpus_file(filename):
    corpus_dir = os.path.abspath(Config().corpus_dir)
    return send_from_directory(corpus_dir, filename, mimetype="text/plain")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "question is required"}), 400

    try:
        resources = get_resources()
        result = answer_question(
            question, resources["retrieve_fn"], resources["llm"], resources["config"]
        )
        return jsonify(result)
    except Exception:
        app.logger.exception("chat request failed")
        return jsonify({"error": "something went wrong, please try again"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
