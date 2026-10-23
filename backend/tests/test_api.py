def test_health(client):
    response = client.get("/health")
    body = response.get_json()
    assert response.status_code == 200
    assert body["status"] == "ok" and body["service"] == "GreenBuddy" and body["nlp_ready"] is True


def test_topics(client):
    body = client.get("/api/topics").get_json()
    assert body["total_entries"] > 300
    assert {"watering", "pet_safety"} <= {t["topic"] for t in body["topics"]}


def test_chat_valid_question(client):
    response = client.post("/chat", json={"question": "Why are my plant's leaves turning yellow?"})
    body = response.get_json()
    assert response.status_code == 200
    assert body["fallback"] is False
    assert body["matched_id"] == "leaf_yellow"
    assert isinstance(body["similarity"], float)
    assert "debug" not in body


def test_chat_debug_mode_shows_pipeline(client):
    body = client.post("/chat", json={"question": "yellow leaves", "debug": True}).get_json()
    assert body["debug"]["pipeline"]["stems"]
    assert len(body["debug"]["top_matches"]) == 3


def test_chat_out_of_domain_returns_fallback_with_200(client):
    body = client.post("/chat", json={"question": "Who won the football match?"}).get_json()
    assert body["fallback"] is True and body["topic"] is None


def assert_error(response, status, code):
    assert response.status_code == status
    assert response.get_json()["error"]["code"] == code


def test_missing_question(client):
    assert_error(client.post("/chat", json={}), 400, "MISSING_QUESTION")


def test_empty_question(client):
    assert_error(client.post("/chat", json={"question": "   "}), 400, "INVALID_QUESTION")


def test_non_string_question(client):
    assert_error(client.post("/chat", json={"question": 123}), 400, "INVALID_QUESTION")


def test_too_long_question(client):
    assert_error(client.post("/chat", json={"question": "a" * 501}), 400, "QUESTION_TOO_LONG")


def test_malformed_json(client):
    response = client.post("/chat", data="{not json", content_type="application/json")
    assert_error(response, 400, "INVALID_JSON")


def test_json_list_is_rejected(client):
    assert_error(client.post("/chat", json=["question"]), 400, "INVALID_JSON")


def test_wrong_content_type(client):
    assert_error(client.post("/chat", data="question=hi"), 415, "UNSUPPORTED_MEDIA_TYPE")


def test_unknown_route_returns_json_404(client):
    assert_error(client.get("/nope"), 404, "NOT_FOUND")


def test_wrong_method_returns_json_405(client):
    # GET is routed to the static-file handler (404), so check another verb.
    assert_error(client.put("/chat", json={"question": "hi"}), 405, "METHOD_NOT_ALLOWED")


def test_internal_error_is_hidden(client, bot, monkeypatch):
    def explode(*args, **kwargs):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(bot, "answer", explode)
    response = client.post("/chat", json={"question": "water"})
    assert_error(response, 500, "INTERNAL_ERROR")
    assert "secret" not in response.get_data(as_text=True)


def test_home_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"GreenBuddy" in response.data


def test_cors_allows_only_local_origins(client):
    allowed = client.get("/health", headers={"Origin": "http://localhost:5001"})
    blocked = client.get("/health", headers={"Origin": "http://evil.example"})
    assert allowed.headers.get("Access-Control-Allow-Origin") == "http://localhost:5001"
    assert "Access-Control-Allow-Origin" not in blocked.headers
