/** Display helpers. Null/undefined/NaN always render as an em dash — never a made-up number. */

export const DASH = '—';

export function isNum(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v);
}

export function fmtNum(v: unknown, digits = 0): string {
  if (!isNum(v)) return DASH;
  return v.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

export function fmtInt(v: unknown): string {
  return isNum(v) ? Math.round(v).toLocaleString() : DASH;
}

export function fmtPct(v: unknown, digits = 1): string {
  return isNum(v) ? `${(v * 100).toFixed(digits)}%` : DASH;
}

export function fmtRatio(v: unknown, digits = 3): string {
  return isNum(v) ? v.toFixed(digits) : DASH;
}

export function fmtMs(v: unknown): string {
  if (!isNum(v)) return DASH;
  if (v >= 10000) return `${(v / 1000).toFixed(1)} s`;
  if (v >= 1000) return `${(v / 1000).toFixed(2)} s`;
  return `${v.toFixed(v < 10 ? 1 : 0)} ms`;
}

export function fmtSeconds(v: unknown): string {
  if (!isNum(v)) return DASH;
  if (v >= 60) return `${Math.floor(v / 60)}m ${Math.round(v % 60)}s`;
  return `${v.toFixed(v < 10 ? 2 : 1)} s`;
}

export function fmtUsd(v: unknown): string {
  if (!isNum(v)) return DASH;
  return v < 0.01 && v > 0 ? `$${v.toFixed(4)}` : `$${v.toFixed(2)}`;
}

export function fmtBytes(v: unknown): string {
  if (!isNum(v)) return DASH;
  const units = ['B', 'KB', 'MB', 'GB'];
  let n = v;
  let i = 0;
  while (n >= 1024 && i < units.length - 1) {
    n /= 1024;
    i++;
  }
  return `${n.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

export function fmtDate(v: unknown): string {
  if (typeof v !== 'string' || !v) return DASH;
  const d = new Date(v);
  if (Number.isNaN(d.getTime())) return v;
  return d.toLocaleString(undefined, { year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

export function fmtRelative(v: unknown): string {
  if (typeof v !== 'string' || !v) return DASH;
  const d = new Date(v);
  const t = d.getTime();
  if (Number.isNaN(t)) return v;
  const s = Math.round((Date.now() - t) / 1000);
  if (s < 0) return fmtDate(v);
  if (s < 60) return 'just now';
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  if (s < 86400 * 7) return `${Math.floor(s / 86400)} d ago`;
  return fmtDate(v);
}

export function humanize(s: string | null | undefined): string {
  if (!s) return DASH;
  const t = s.replace(/_/g, ' ');
  return t.charAt(0).toUpperCase() + t.slice(1);
}

export function splitList(s: string): string[] {
  return s
    .split(/[,\n]/)
    .map((x) => x.trim())
    .filter(Boolean);
}
