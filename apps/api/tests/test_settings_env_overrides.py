"""KALIMBA_SETTINGS_OVERRIDES (ablation observatory 用 env フック) の mechanism テスト。

import 時に一度だけ読む設計なので、テストは純関数 _apply_env_overrides を直接叩く。
"""

from __future__ import annotations

from app.transcription.settings import RecognizerSettings, _apply_env_overrides


def test_env_override_flips_flag_and_converts_gates(monkeypatch):
    monkeypatch.setenv(
        "KALIMBA_SETTINGS_OVERRIDES",
        '{"use_onset_gate": false, "disabled_gates": ["semitone-leakage"], "unknown_key": 1}',
    )
    settings = _apply_env_overrides(RecognizerSettings())
    assert settings.use_onset_gate is False
    assert settings.disabled_gates == frozenset({"semitone-leakage"})
    # 未知キーは無視される (将来のフィールド削除に対して安全)
    assert not hasattr(settings, "unknown_key")


def test_env_override_noop_without_env_or_on_bad_json(monkeypatch):
    monkeypatch.delenv("KALIMBA_SETTINGS_OVERRIDES", raising=False)
    base = RecognizerSettings()
    assert _apply_env_overrides(base) is base
    monkeypatch.setenv("KALIMBA_SETTINGS_OVERRIDES", "{broken json")
    assert _apply_env_overrides(base) is base
