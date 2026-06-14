import { act, renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { useAudioPlayback } from "@/hooks/useAudioPlayback";
import { ScoreEvent } from "@/lib/types";

function buildEvents(): ScoreEvent[] {
  return [
    {
      id: "evt-1",
      startBeat: 1,
      durationBeat: 1,
      startTimeSec: 0.5,
      notes: [
        { key: 1, pitchClass: "C", octave: 4, labelDoReMi: "ド", labelNumber: "1", frequency: 261.63 },
      ],
      isGlissLike: false,
      gesture: "strict_chord",
    },
    {
      id: "evt-2",
      startBeat: 2,
      durationBeat: 1,
      startTimeSec: 2.0,
      notes: [
        { key: 2, pitchClass: "E", octave: 4, labelDoReMi: "ミ", labelNumber: "3", frequency: 329.63 },
      ],
      isGlissLike: false,
      gesture: "strict_chord",
    },
    {
      id: "evt-3",
      startBeat: 3,
      durationBeat: 1,
      startTimeSec: 3.5,
      notes: [
        { key: 3, pitchClass: "G", octave: 4, labelDoReMi: "ソ", labelNumber: "5", frequency: 392 },
      ],
      isGlissLike: false,
      gesture: "strict_chord",
    },
  ];
}

// jsdom は HTMLAudioElement.play() / currentTime を実装しないので、必要な面だけ持つ
// 軽量モックを audioRef.current に差し込む。
function createMockAudio() {
  return {
    currentTime: 0,
    play: vi.fn(),
    pause: vi.fn(),
  };
}

describe("useAudioPlayback", () => {
  it("seekToEvent seeks just before the onset and plays, marking the active event", () => {
    const events = buildEvents();
    const onActiveEventChange = vi.fn();
    const { result } = renderHook(() => useAudioPlayback(events, onActiveEventChange));

    const audio = createMockAudio();
    result.current.audioRef.current = audio as unknown as HTMLAudioElement;

    act(() => {
      result.current.seekToEvent("evt-2");
    });

    // 0.15s の lead を引いた位置に頭出し。
    expect(audio.currentTime).toBeCloseTo(2.0 - 0.15, 5);
    expect(audio.play).toHaveBeenCalledTimes(1);
    expect(onActiveEventChange).toHaveBeenCalledWith("evt-2");
  });

  it("seekToEvent with play:false seeks exactly to the onset without playing", () => {
    const events = buildEvents();
    const onActiveEventChange = vi.fn();
    const { result } = renderHook(() => useAudioPlayback(events, onActiveEventChange));

    const audio = createMockAudio();
    result.current.audioRef.current = audio as unknown as HTMLAudioElement;

    act(() => {
      result.current.seekToEvent("evt-3", { play: false });
    });

    expect(audio.currentTime).toBeCloseTo(3.5, 5);
    expect(audio.play).not.toHaveBeenCalled();
    expect(onActiveEventChange).toHaveBeenCalledWith("evt-3");
  });

  it("section playback pauses at the next event's onset", () => {
    const events = buildEvents();
    const onActiveEventChange = vi.fn();
    const { result } = renderHook(() => useAudioPlayback(events, onActiveEventChange));

    const audio = createMockAudio();
    result.current.audioRef.current = audio as unknown as HTMLAudioElement;

    act(() => {
      result.current.seekToEvent("evt-1");
    });

    // evt-1 (0.5s) の区間は次イベント evt-2 (2.0s) で止まる。まだ手前なら止まらない。
    audio.currentTime = 1.9;
    act(() => {
      result.current.handleTimeUpdate();
    });
    expect(audio.pause).not.toHaveBeenCalled();

    // 区間終端を越えたら pause。
    audio.currentTime = 2.05;
    act(() => {
      result.current.handleTimeUpdate();
    });
    expect(audio.pause).toHaveBeenCalledTimes(1);
  });

  it("handleTimeUpdate follows playback position and emits the current event only on change", () => {
    const events = buildEvents();
    const onActiveEventChange = vi.fn();
    const { result } = renderHook(() => useAudioPlayback(events, onActiveEventChange));

    const audio = createMockAudio();
    result.current.audioRef.current = audio as unknown as HTMLAudioElement;

    audio.currentTime = 0.6; // evt-1 領域
    act(() => result.current.handleTimeUpdate());
    expect(onActiveEventChange).toHaveBeenLastCalledWith("evt-1");

    // 同じ event 内では再通知しない。
    onActiveEventChange.mockClear();
    audio.currentTime = 1.0;
    act(() => result.current.handleTimeUpdate());
    expect(onActiveEventChange).not.toHaveBeenCalled();

    // 次 event に入ったら通知。
    audio.currentTime = 2.1;
    act(() => result.current.handleTimeUpdate());
    expect(onActiveEventChange).toHaveBeenLastCalledWith("evt-2");
  });

  it("a user-initiated seek clears the section stop, but a programmatic seek keeps it", () => {
    const events = buildEvents();
    const onActiveEventChange = vi.fn();
    const { result } = renderHook(() => useAudioPlayback(events, onActiveEventChange));

    const audio = createMockAudio();
    result.current.audioRef.current = audio as unknown as HTMLAudioElement;

    act(() => {
      result.current.seekToEvent("evt-1");
    });

    // seekToEvent が立てた programmatic フラグを onSeeking が消費する → stopAt 維持。
    act(() => result.current.handleSeeking());
    audio.currentTime = 2.05;
    act(() => result.current.handleTimeUpdate());
    expect(audio.pause).toHaveBeenCalledTimes(1);

    // 改めて区間再生をセットしてから、ユーザー seek (programmatic フラグ無し) を起こすと解除。
    audio.pause.mockClear();
    act(() => {
      result.current.seekToEvent("evt-1");
    });
    act(() => result.current.handleSeeking()); // programmatic 消費
    act(() => result.current.handleSeeking()); // 2 回目 = ユーザー seek → stopAt 解除
    audio.currentTime = 2.05;
    act(() => result.current.handleTimeUpdate());
    expect(audio.pause).not.toHaveBeenCalled();
  });
});
