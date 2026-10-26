const EMOJI = ['📘', '🧪', '📊', '🧠', '🛰️', '⚙️', '🔬', '📚', '🧭', '💡', '🌿', '🗂️'];

function hash(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

/** A stable soft colour and emoji per notebook id. */
export function notebookStyle(id: string): { emoji: string; hue: number } {
  const h = hash(id);
  return { emoji: EMOJI[h % EMOJI.length], hue: (h >>> 8) % 360 };
}
