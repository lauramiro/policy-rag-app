import app as app_module


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_index_renders(client):
    response = client.get("/")

    assert response.status_code == 200
    assert b"Company Policy Assistant" in response.data


def test_chat_rejects_empty_question(client):
    response = client.post("/chat", json={"question": "   "})

    assert response.status_code == 400


def test_chat_returns_answer_and_citations(client, monkeypatch):
    monkeypatch.setattr(
        app_module, "get_resources",
        lambda: {"config": None, "retrieve_fn": None, "llm": None},
    )
    monkeypatch.setattr(
        app_module, "answer_question",
        lambda *args, **kwargs: {
            "answer": "You get 15 PTO days per year.",
            "citations": [{"doc_id": "pto-policy", "title": "PTO Policy", "snippet": "..."}],
        },
    )

    response = client.post("/chat", json={"question": "How many PTO days do I get?"})

    assert response.status_code == 200
    body = response.get_json()
    assert body["answer"] == "You get 15 PTO days per year."
    assert body["citations"][0]["doc_id"] == "pto-policy"


def test_chat_handles_internal_errors_gracefully(client, monkeypatch):
    monkeypatch.setattr(
        app_module, "get_resources",
        lambda: {"config": None, "retrieve_fn": None, "llm": None},
    )

    def _boom(*args, **kwargs):
        raise RuntimeError("groq is down")

    monkeypatch.setattr(app_module, "answer_question", _boom)

    response = client.post("/chat", json={"question": "How many PTO days do I get?"})

    assert response.status_code == 500
    assert "error" in response.get_json()
