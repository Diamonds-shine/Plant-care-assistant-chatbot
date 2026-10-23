"""GreenBuddy Flask server: serves the web page and the chatbot API.

Run (macOS / Linux):   python3 app.py
Run (Windows):         python app.py
Then open http://127.0.0.1:5001 in a browser.

Endpoints
  GET  /                                  - the chat web page (frontend/index.html)
  GET  /health       (alias /api/health)  - is the server and NLP engine ready?
  GET  /api/topics                        - supported topics with counts
  POST /chat         (alias /api/chat)    - {"question": "..."} -> answer JSON
"""

from __future__ import annotations

import logging

from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

import config
from nlp.chatbot import PlantCareChatbot, RetrievalResult
from nlp.preprocess import InvalidInputError, nltk_status

logger = logging.getLogger("greenbuddy")


def error_response(status: int, code: str, message: str):
    return jsonify({"error": {"code": code, "message": message}}), status


def result_to_json(result: RetrievalResult, include_debug: bool) -> dict:
    body = {
        "answer": result.answer,
        "topic": result.topic,
        "topic_label": result.topic_label,
        "matched_question": result.matched_question,
        "matched_id": result.matched_id,
        "similarity": round(result.similarity, 3),
        "confidence": result.confidence,
        "fallback": result.fallback,
        "fallback_reason": result.fallback_reason,
        "source": result.source,
        "related_questions": result.related_questions,
    }
    if include_debug:
        body["debug"] = {
            "pipeline": result.processed.to_dict() if result.processed else None,
            "top_matches": [
                {"id": m.entry_id, "question": m.question, "topic": m.topic, "similarity": round(m.score, 3)}
                for m in result.top_matches
            ],
            "threshold": config.LOW_CONFIDENCE_THRESHOLD,
        }
    return body


def create_app(chatbot: PlantCareChatbot | None = None) -> Flask:
    """Application factory. Tests can pass in their own chatbot instance."""
    # The plain HTML/CSS/JS frontend lives in ../frontend and is served as static files.
    app = Flask(__name__, static_folder=str(config.FRONTEND_DIR), static_url_path="")
    app.json.sort_keys = False
    CORS(app, origins=config.CORS_ORIGINS)

    # Build the TF-IDF index once, at start-up, not on every request.
    app.extensions["greenbuddy"] = chatbot or PlantCareChatbot()

    def bot() -> PlantCareChatbot:
        return app.extensions["greenbuddy"]

    @app.get("/")
    def index():
        return app.send_static_file("index.html")

    @app.get("/health")
    @app.get("/api/health")
    def health():
        engine = bot()
        return jsonify({
            "status": "ok",
            "service": "GreenBuddy",
            "nlp_ready": engine is not None,
            "entries": len(engine.entries),
            "sources": engine.source_counts(),
            "nltk": nltk_status(),
        })

    @app.get("/api/topics")
    def topics():
        engine = bot()
        return jsonify({"topics": engine.topics(), "total_entries": len(engine.entries)})

    @app.post("/chat")
    @app.post("/api/chat")
    def chat():
        if not request.is_json:
            return error_response(415, "UNSUPPORTED_MEDIA_TYPE", "Send the question as JSON (Content-Type: application/json).")
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return error_response(400, "INVALID_JSON", "The request body must be a JSON object.")
        if "question" not in payload:
            return error_response(400, "MISSING_QUESTION", "Please include a 'question' field.")
        try:
            result = bot().answer(payload["question"])
        except InvalidInputError as exc:
            return error_response(400, exc.code, exc.message)
        # Log the outcome only, not the user's text.
        logger.info("chat: fallback=%s topic=%s score=%.3f", result.fallback, result.topic, result.similarity)
        return jsonify(result_to_json(result, include_debug=payload.get("debug") is True))

    @app.errorhandler(HTTPException)
    def handle_http_error(exc: HTTPException):
        code = (exc.name or "error").upper().replace(" ", "_")
        return error_response(exc.code or 500, code, exc.description or "Request failed.")

    @app.errorhandler(Exception)
    def handle_unexpected_error(exc: Exception):
        # Full details go to the server log; the client gets a generic message.
        logger.exception("Unhandled error while processing %s", request.path)
        return error_response(500, "INTERNAL_ERROR", "Something went wrong on our side. Please try again.")

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    application = create_app()
    print(f"GreenBuddy API running on http://{config.HOST}:{config.PORT}")
    application.run(host=config.HOST, port=config.PORT, debug=False)
