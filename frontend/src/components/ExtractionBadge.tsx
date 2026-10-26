import { DASH } from '../utils/format';
import { Badge } from './ui';

/** How a page/passage's text was obtained: the PDF's text layer or OCR. */
export function ExtractionBadge({ method }: { method: string | null | undefined }) {
  if (!method) return <span className="muted">{DASH}</span>;
  if (method.startsWith('ocr:')) return <Badge tone="warning" title={`Text produced by OCR (${method})`}>OCR</Badge>;
  if (method === 'text_layer') return <Badge tone="success" title="Text taken from the PDF's text layer">text layer</Badge>;
  return <Badge>{method}</Badge>;
}
