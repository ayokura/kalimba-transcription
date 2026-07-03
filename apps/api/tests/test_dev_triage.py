"""Dev triage endpoint (/api/dev/triage) — temporary, 第 2 期 S1.

/debug/triage ページの供給源。transactions_triage.py の出力 JSON に live の
review status を重ねるだけの薄い endpoint なので、404 経路と status merge
のみを pin する。
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.main import app


def test_dev_triage_404_without_summary(tmp_path, monkeypatch):
    monkeypatch.setenv("KALIMBA_DATA_DIR", str(tmp_path))
    client = TestClient(app)
    response = client.get("/api/dev/triage")
    assert response.status_code == 404
    assert "transactions_triage.py" in response.json()["detail"]


def test_dev_triage_merges_live_review_status(tmp_path, monkeypatch):
    monkeypatch.setenv("KALIMBA_DATA_DIR", str(tmp_path))
    tx_dir = tmp_path / "transactions" / "tx-live"
    tx_dir.mkdir(parents=True)
    (tx_dir / "review_status.json").write_text(
        json.dumps({"version": 1, "status": "unusable"}), encoding="utf-8"
    )

    summary = {
        "generatedAt": "2026-07-04T00:00:00+00:00",
        "recognizerFingerprint": "test",
        "totals": {
            "transactionDirs": 1,
            "uniqueRecordings": 1,
            "withGt": 0,
            "statusCounts": {},
        },
        "recordings": [
            {
                "sha16": "abc",
                "primaryTx": "tx-live",
                "duplicateTxs": ["tx-missing"],
                # transactions_triage.py が書いた時点の status は stale でよい
                "reviewStatuses": {"tx-live": None},
                "score": 0,
                "signals": [],
            }
        ],
    }
    (tmp_path / "triage_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    client = TestClient(app)
    response = client.get("/api/dev/triage")
    assert response.status_code == 200
    recording = response.json()["recordings"][0]
    # live の review_status.json が stale なスナップショットを上書きする
    assert recording["reviewStatuses"]["tx-live"] == "unusable"
    assert recording["reviewStatuses"]["tx-missing"] is None
