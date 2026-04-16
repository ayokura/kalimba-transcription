import json
import re

from app.storage import (
    generate_transaction_id,
    get_transaction_dir,
    load_audio_path,
    load_response,
    save_transaction,
)

UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def test_generate_transaction_id_is_uuid():
    tid = generate_transaction_id()
    assert UUID_RE.match(tid), f"Not a valid UUID: {tid}"


def test_generate_transaction_id_unique():
    ids = {generate_transaction_id() for _ in range(100)}
    assert len(ids) == 100


def test_save_and_load_response():
    tid = generate_transaction_id()
    audio = b"RIFF" + b"\x00" * 100
    request_params = {"tuning": {"id": "test"}, "debug": False}
    response = {"transactionId": tid, "events": []}

    save_transaction(tid, audio, request_params, response, None)

    loaded = load_response(tid)
    assert loaded == response


def test_save_with_debug():
    tid = generate_transaction_id()
    audio = b"RIFF" + b"\x00" * 100
    request_params = {"tuning": {"id": "test"}, "debug": True}
    response = {"transactionId": tid, "events": [], "debug": {"info": "test"}}
    debug_dict = {"info": "test"}

    save_transaction(tid, audio, request_params, response, debug_dict)

    tx_dir = get_transaction_dir(tid)
    assert (tx_dir / "debug.json").exists()
    assert json.loads((tx_dir / "debug.json").read_text()) == debug_dict


def test_load_audio_path():
    tid = generate_transaction_id()
    audio = b"RIFF" + b"\x00" * 100

    save_transaction(tid, audio, {}, {}, None)

    path = load_audio_path(tid)
    assert path is not None
    assert path.read_bytes() == audio


def test_load_response_nonexistent():
    assert load_response("00000000-0000-0000-0000-000000000000") is None


def test_load_audio_path_nonexistent():
    assert load_audio_path("00000000-0000-0000-0000-000000000000") is None
