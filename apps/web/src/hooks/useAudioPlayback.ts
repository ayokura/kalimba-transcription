"use client";

import { useCallback, useRef } from "react";

import { findEventById, findEventIdAtSec } from "@/lib/eventTiming";
import { ScoreEvent } from "@/lib/types";

// ReviewEditor の auditionEvent と同じ区間再生の挙動を共有する定数。
// onset の少し手前から鳴らし、次イベント (なければ最大 4 秒) で止める。
const AUDITION_LEAD_SEC = 0.15;
const AUDITION_MAX_SEC = 4.0;

export type SeekToEventOptions = {
  /** seek 後に play() するか (default: true)。区間再生せず頭出しだけしたい時は false。 */
  play?: boolean;
};

export type AudioPlaybackController = {
  audioRef: React.MutableRefObject<HTMLAudioElement | null>;
  /** 指定 event の startTimeSec へ頭出しし、必要なら次イベントまでの区間を再生する。 */
  seekToEvent: (eventId: string, options?: SeekToEventOptions) => void;
  /** <audio onTimeUpdate>。区間終端での停止と activeEventId 追従を行う。 */
  handleTimeUpdate: () => void;
  /** <audio onSeeking>。ユーザー自身の seek でのみ区間停止を解除する。 */
  handleSeeking: () => void;
};

/**
 * Review workspace の event-level audition を司るフック。
 * ReviewEditor.tsx の audioRef + stopAtRef + programmaticSeekRef パターンを共有化したもの。
 *
 * @param events 表示中の ScoreEvent 配列 (時間昇順を仮定)
 * @param onActiveEventChange timeupdate で現在位置の event が変わった時に呼ばれる
 */
export function useAudioPlayback(
  events: ScoreEvent[],
  onActiveEventChange: (eventId: string) => void,
): AudioPlaybackController {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const stopAtRef = useRef<number | null>(null);
  const programmaticSeekRef = useRef(false);
  // timeupdate ごとに親へ通知すると無駄な再レンダーが多いので、直近の event id を記憶し
  // 変化した時だけ onActiveEventChange を呼ぶ。
  const lastEmittedRef = useRef<string | null>(null);

  const seekToEvent = useCallback(
    (eventId: string, options?: SeekToEventOptions) => {
      const audio = audioRef.current;
      const event = findEventById(events, eventId);
      if (!audio || !event) return;

      const shouldPlay = options?.play ?? true;
      if (shouldPlay) {
        // 次イベント (なければ最大 AUDITION_MAX_SEC) で止める区間再生。
        const followers = events.filter((e) => e.startTimeSec > event.startTimeSec + 0.01);
        const nextStart =
          followers.length > 0 ? followers[0].startTimeSec : event.startTimeSec + AUDITION_MAX_SEC;
        stopAtRef.current = Math.min(nextStart, event.startTimeSec + AUDITION_MAX_SEC);
        // currentTime 代入も onSeeking を発火させるため、programmatic seek を
        // マークして stopAt の解除対象から除外する (ユーザー操作の seek のみ解除)。
        programmaticSeekRef.current = true;
        audio.currentTime = Math.max(0, event.startTimeSec - AUDITION_LEAD_SEC);
        void audio.play();
      } else {
        // 頭出しのみ。区間再生はしないので stopAt は持たない。
        stopAtRef.current = null;
        programmaticSeekRef.current = true;
        audio.currentTime = Math.max(0, event.startTimeSec);
      }
      lastEmittedRef.current = eventId;
      onActiveEventChange(eventId);
    },
    [events, onActiveEventChange],
  );

  const handleTimeUpdate = useCallback(() => {
    const audio = audioRef.current;
    if (!audio) return;

    if (stopAtRef.current !== null && audio.currentTime >= stopAtRef.current) {
      audio.pause();
      stopAtRef.current = null;
    }

    const next = findEventIdAtSec(events, audio.currentTime);
    if (next !== null && next !== lastEmittedRef.current) {
      lastEmittedRef.current = next;
      onActiveEventChange(next);
    }
  }, [events, onActiveEventChange]);

  const handleSeeking = useCallback(() => {
    // seekToEvent 由来の programmatic seek では stopAt を維持し、
    // ユーザーが自分でシークした時だけ区間再生を解除する。
    if (programmaticSeekRef.current) {
      programmaticSeekRef.current = false;
      return;
    }
    stopAtRef.current = null;
  }, []);

  return { audioRef, seekToEvent, handleTimeUpdate, handleSeeking };
}
