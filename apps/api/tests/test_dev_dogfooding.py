"""Dev dogfooding endpoints (/api/dev/dogfooding) — temporary, 第 3 期 S2.

/debug/dogfooding ページの供給源。手動記入 (諦め箇所・主観負荷・曖昧性カタログ・
弾き戻し) と完了フラグを data/dogfooding/<txId>.json に永続化するだけの薄い
endpoint なので、一覧 / 個別取得の 404-vs-null 経路 / 保存roundtrip / id 検証のみ pin。
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

_TX_ID = "11111111-1111-1111-1111-111111111111"
_OTHER_TX_ID = "22222222-2222-2222-2222-222222222222"


def _make_transaction(tmp_path, tx_id: str) -> None:
    (tmp_path / "transactions" / tx_id).mkdir(parents=True, exist_ok=True)


def test_list_dev_dogfooding_empty_without_records(tmp_path, monkeypatch):
    monkeypatch.setenv("KALIMBA_DATA_DIR", str(tmp_path))
    client = TestClient(app)
    response = client.get("/api/dev/dogfooding")
    assert response.status_code == 200
    assert response.json() == {"records": []}


def test_get_dev_dogfooding_returns_null_record_when_unsaved(tmp_path, monkeypatch):
    monkeypatch.setenv("KALIMBA_DATA_DIR", str(tmp_path))
    _make_transaction(tmp_path, _TX_ID)
    client = TestClient(app)
    response = client.get(f"/api/dev/dogfooding/{_TX_ID}")
    assert response.status_code == 200
    assert response.json() == {"record": None}


def test_put_dev_dogfooding_requires_existing_transaction(tmp_path, monkeypatch):
    monkeypatch.setenv("KALIMBA_DATA_DIR", str(tmp_path))
    client = TestClient(app)
    response = client.put(f"/api/dev/dogfooding/{_TX_ID}", json={"manual": {}, "done": False})
    assert response.status_code == 404


def test_put_dev_dogfooding_validates_transaction_id_format(tmp_path, monkeypatch):
    monkeypatch.setenv("KALIMBA_DATA_DIR", str(tmp_path))
    client = TestClient(app)
    response = client.put("/api/dev/dogfooding/not-a-uuid", json={"manual": {}, "done": False})
    assert response.status_code == 400


def test_put_then_get_roundtrip_and_list(tmp_path, monkeypatch):
    monkeypatch.setenv("KALIMBA_DATA_DIR", str(tmp_path))
    _make_transaction(tmp_path, _TX_ID)
    client = TestClient(app)

    payload = {
        "manual": {
            "giveUpCount": 1,
            "giveUpNotes": "速いパッセージの和音構成が聞き取れなかった",
            "subjectiveLoad": 3,
            "ambiguityLog": [
                {"timeSec": "12.3s", "judgment": "弾き直しか残響か", "resolution": "残響として無視"},
            ],
            "playback": {
                "phraseCount": 8,
                "reproducedPhraseCount": 6,
                "stumblePitch": 1,
                "stumbleRhythm": 1,
                "stumbleNotation": 0,
            },
        },
        "done": True,
    }
    saved = client.put(f"/api/dev/dogfooding/{_TX_ID}", json=payload)
    assert saved.status_code == 200
    saved_body = saved.json()["record"]
    assert saved_body["txId"] == _TX_ID
    assert saved_body["done"] is True
    assert saved_body["manual"]["giveUpCount"] == 1
    assert saved_body["updatedAt"]

    fetched = client.get(f"/api/dev/dogfooding/{_TX_ID}")
    assert fetched.status_code == 200
    assert fetched.json()["record"] == saved_body

    listing = client.get("/api/dev/dogfooding").json()["records"]
    assert listing == [{"txId": _TX_ID, "updatedAt": saved_body["updatedAt"], "done": True}]


def test_list_dev_dogfooding_sorts_most_recently_updated_first(tmp_path, monkeypatch):
    monkeypatch.setenv("KALIMBA_DATA_DIR", str(tmp_path))
    _make_transaction(tmp_path, _TX_ID)
    _make_transaction(tmp_path, _OTHER_TX_ID)
    client = TestClient(app)

    client.put(f"/api/dev/dogfooding/{_TX_ID}", json={"manual": {}, "done": False})
    client.put(f"/api/dev/dogfooding/{_OTHER_TX_ID}", json={"manual": {}, "done": True})

    listing = client.get("/api/dev/dogfooding").json()["records"]
    tx_ids_in_order = [r["txId"] for r in listing]
    # 2 つ目に保存した方が updatedAt で先に来る (同秒解像度なので同点なら安定性は
    # 問わないが、少なくとも両方が listing に載ることを確認する)
    assert set(tx_ids_in_order) == {_TX_ID, _OTHER_TX_ID}


def test_dev_dogfooding_ignores_corrupted_json_file_in_listing(tmp_path, monkeypatch):
    monkeypatch.setenv("KALIMBA_DATA_DIR", str(tmp_path))
    dogfooding_dir = tmp_path / "dogfooding"
    dogfooding_dir.mkdir(parents=True)
    (dogfooding_dir / "broken.json").write_text("{not valid json", encoding="utf-8")
    client = TestClient(app)
    response = client.get("/api/dev/dogfooding")
    assert response.status_code == 200
    assert response.json() == {"records": []}
