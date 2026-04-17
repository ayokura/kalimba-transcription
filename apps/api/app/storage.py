from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path


def get_data_dir() -> Path:
    return Path(os.environ.get("KALIMBA_DATA_DIR", "data"))


def get_transactions_dir() -> Path:
    return get_data_dir() / "transactions"


def get_transaction_dir(transaction_id: str) -> Path:
    return get_transactions_dir() / transaction_id


def generate_transaction_id() -> str:
    return str(uuid.uuid4())


def compute_audio_sha256(audio_bytes: bytes) -> str:
    return hashlib.sha256(audio_bytes).hexdigest()


def find_transaction_by_hash_and_tuning(audio_sha256: str, tuning_id: str) -> str | None:
    tx_root = get_transactions_dir()
    if not tx_root.exists():
        return None
    for tx_dir in tx_root.iterdir():
        if not tx_dir.is_dir():
            continue
        request_path = tx_dir / "request.json"
        if not request_path.exists():
            continue
        try:
            data = json.loads(request_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        stored_hash = data.get("audioSha256")
        stored_tuning_id = (data.get("tuning") or {}).get("id")
        if stored_hash == audio_sha256 and stored_tuning_id == tuning_id:
            return tx_dir.name
    return None


def list_recent_transactions(limit: int) -> list[dict]:
    tx_root = get_transactions_dir()
    if not tx_root.exists():
        return []
    entries: list[tuple[float, Path]] = []
    for tx_dir in tx_root.iterdir():
        if not tx_dir.is_dir():
            continue
        audio_path = tx_dir / "audio.wav"
        if not audio_path.exists():
            continue
        entries.append((audio_path.stat().st_mtime, tx_dir))
    entries.sort(reverse=True)
    results: list[dict] = []
    for _, tx_dir in entries[:limit]:
        entry = _summarize_transaction(tx_dir)
        if entry is not None:
            results.append(entry)
    return results


def _summarize_transaction(tx_dir: Path) -> dict | None:
    request_path = tx_dir / "request.json"
    response_path = tx_dir / "response.json"
    if not request_path.exists() or not response_path.exists():
        return None
    try:
        request_data = json.loads(request_path.read_text(encoding="utf-8"))
        response_data = json.loads(response_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    tuning = request_data.get("tuning") or {}
    audio_path = tx_dir / "audio.wav"
    return {
        "transactionId": tx_dir.name,
        "createdAt": audio_path.stat().st_mtime,
        "tuningId": tuning.get("id"),
        "tuningName": tuning.get("name"),
        "eventCount": len(response_data.get("events") or []),
        "audioSha256": request_data.get("audioSha256"),
    }


def save_transaction(
    transaction_id: str,
    audio_bytes: bytes,
    request_params: dict,
    response_dict: dict,
    debug_dict: dict | None,
) -> None:
    tx_dir = get_transaction_dir(transaction_id)
    tx_dir.mkdir(parents=True, exist_ok=True)
    (tx_dir / "audio.wav").write_bytes(audio_bytes)
    (tx_dir / "request.json").write_text(
        json.dumps(request_params, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (tx_dir / "response.json").write_text(
        json.dumps(response_dict, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if debug_dict is not None:
        (tx_dir / "debug.json").write_text(
            json.dumps(debug_dict, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def load_response(transaction_id: str) -> dict | None:
    response_path = get_transaction_dir(transaction_id) / "response.json"
    if not response_path.exists():
        return None
    return json.loads(response_path.read_text(encoding="utf-8"))


def load_audio_path(transaction_id: str) -> Path | None:
    audio_path = get_transaction_dir(transaction_id) / "audio.wav"
    return audio_path if audio_path.exists() else None


def save_memo(transaction_id: str, memo: str) -> None:
    tx_dir = get_transaction_dir(transaction_id)
    tx_dir.mkdir(parents=True, exist_ok=True)
    (tx_dir / "memo.txt").write_text(memo, encoding="utf-8")


def load_memo(transaction_id: str) -> str | None:
    memo_path = get_transaction_dir(transaction_id) / "memo.txt"
    if not memo_path.exists():
        return None
    return memo_path.read_text(encoding="utf-8")


def transaction_exists(transaction_id: str) -> bool:
    return get_transaction_dir(transaction_id).exists()
