import { fmtBytes } from './format';

export function isPdf(f: File): boolean {
  return f.type === 'application/pdf' || f.name.toLowerCase().endsWith('.pdf');
}

/** Client-side pre-check; the server re-validates (and also rejects encrypted / unreadable PDFs). */
export function rejectReason(f: File, maxUploadMb: number | undefined): string | undefined {
  if (!isPdf(f)) return 'Only PDF files are supported.';
  if (f.size === 0) return 'The file is empty.';
  if (maxUploadMb && f.size > maxUploadMb * 1024 * 1024) return `File is ${fmtBytes(f.size)}; the limit is ${maxUploadMb} MB.`;
  return undefined;
}
