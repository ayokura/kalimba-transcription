from __future__ import annotations

import json
import os
import uuid
from pathlib import Path


def get_data_dir() -> Path:
    return Path(os.environ.get("KALIMBA_DATA_DIR", "data"))


def get_transaction_dir(transaction_id: str) -> Path:
    return get_data_dir() / "transactions" / transaction_id


def generate_transaction_id() -> str:
    return str(uuid.uuid4())


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
