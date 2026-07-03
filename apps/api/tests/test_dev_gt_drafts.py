"""Dev GT-draft endpoints (/api/dev/gt-drafts) — temporary, 第 2 期 S2.

/debug/gt-review ページの供給源。gt_draft.py の rows.json を列挙して verdict を
重ねるだけの薄い endpoint なので、404 経路 / verdict roundtrip / id 検証のみ pin。
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.main import app


def _write_rows(tmp_path, tx8: str) -> None:
    drafts_dir = tmp_path / "gt_drafts"
    drafts_dir.mkdir(exist_ok=True)
    (drafts_dir / f"{tx8}.rows.json").write_text(
        json.dumps(
            {
                "txId": f"{tx8}-full-transaction-id",
                "tx8": tx8,
                "durationSec": 10.0,
                "tuningNotes": ["C4", "D4"],
                "rows": [
                    {
                        "index": 1,
                        "timeSec": 1.0,
                        "top": [{"note": "C4", "score": 1.0, "share": 0.9}],
                        "flag": "ok",
                        "draftNotes": ["C4"],
                        "comment": "",
                    }
                ],
                "unplacedExpected": [],
            }
        ),
        encoding="utf-8",
    )


def test_dev_gt_drafts_404_without_drafts(tmp_path, monkeypatch):
    monkeypatch.setenv("KALIMBA_DATA_DIR", str(tmp_path))
    client = TestClient(app)
    response = client.get("/api/dev/gt-drafts")
    assert response.status_code == 404
    assert "gt_draft.py" in response.json()["detail"]


def test_dev_gt_drafts_lists_rows_and_merges_verdict(tmp_path, monkeypatch):
    monkeypatch.setenv("KALIMBA_DATA_DIR", str(tmp_path))
    _write_rows(tmp_path, "aabbccdd")
    client = TestClient(app)

    response = client.get("/api/dev/gt-drafts")
    assert response.status_code == 200
    drafts = response.json()["drafts"]
    assert len(drafts) == 1
    assert drafts[0]["tx8"] == "aabbccdd"
    assert drafts[0]["verdict"] is None

    verdict = {
        "rows": {"1": {"decision": "fix", "notes": ["D4"]}},
        "unplaced": {"2": {"decision": "place", "timeSec": 3.5}},
        "done": True,
    }
    saved = client.put("/api/dev/gt-drafts/aabbccdd/verdict", json=verdict)
    assert saved.status_code == 200
    assert saved.json()["verdict"]["savedAt"]

    merged = client.get("/api/dev/gt-drafts").json()["drafts"][0]["verdict"]
    assert merged["rows"]["1"] == {"decision": "fix", "notes": ["D4"]}
    assert merged["done"] is True


def test_dev_gt_draft_verdict_validates_id(tmp_path, monkeypatch):
    monkeypatch.setenv("KALIMBA_DATA_DIR", str(tmp_path))
    _write_rows(tmp_path, "aabbccdd")
    client = TestClient(app)
    assert (
        client.put("/api/dev/gt-drafts/../evil/verdict", json={"rows": {}}).status_code
        in (400, 404)
    )
    assert client.put("/api/dev/gt-drafts/ZZZZZZZZ/verdict", json={"rows": {}}).status_code == 400
    assert client.put("/api/dev/gt-drafts/00000000/verdict", json={"rows": {}}).status_code == 404
