export function buildGestureLabel(gesture: string) {
  if (gesture === "strict_chord") return "同時和音";
  if (gesture === "slide_chord") return "スライド和音";
  if (gesture === "arpeggio") return "アルペジオ";
  if (gesture === "separated_notes") return "単音列";
  return gesture;
}
