from types import SimpleNamespace

import scripts.ingest as ingest_script


def test_main_reports_ingested_chunk_count(monkeypatch, capsys):
    fake_vectorstore = SimpleNamespace(_collection=SimpleNamespace(count=lambda: 42))
    monkeypatch.setattr(ingest_script, "build_vectorstore", lambda *a, **k: fake_vectorstore)

    ingest_script.main()

    captured = capsys.readouterr()
    assert "42" in captured.out
