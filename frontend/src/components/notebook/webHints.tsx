import { Link } from 'react-router-dom';
import { ApiError } from '../../services/api';

/** True when the server says web search has no search service connected (400 "Web search needs a search service…"). */
export function isNoSearchError(e: unknown): boolean {
  return e instanceof ApiError && e.status === 400 && /search service/i.test(e.detail);
}

/** Friendly replacement for the "no search service" error, with a link to connect one. */
export function NoSearchHint() {
  return (
    <div className="nb-connect-hint" role="status">
      Web search needs a search service. <Link to="/settings/connect">Connect Tavily, Brave or Exa</Link> in Settings → Connect AI — it takes about a minute.
    </div>
  );
}

/** "www.example.com" → "example.com"; returns the input when it is not a URL. */
export function domainOf(url: string | null | undefined): string {
  if (!url) return '';
  try {
    return new URL(url).hostname.replace(/^www\./, '');
  } catch {
    return url;
  }
}

export function GlobeIcon({ className = 'nb-file-icon nb-globe-icon' }: { className?: string }) {
  return (
    <svg className={className} width="18" height="18" viewBox="0 0 24 24" aria-hidden="true">
      <circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" strokeWidth="1.6" />
      <path d="M3 12h18M12 3c2.5 2.6 3.8 5.6 3.8 9s-1.3 6.4-3.8 9c-2.5-2.6-3.8-5.6-3.8-9S9.5 5.6 12 3z" fill="none" stroke="currentColor" strokeWidth="1.6" />
    </svg>
  );
}

export function FileIcon() {
  return (
    <svg className="nb-file-icon" width="18" height="18" viewBox="0 0 24 24" aria-hidden="true">
      <path d="M6 2h8l5 5v13a2 2 0 01-2 2H6a2 2 0 01-2-2V4a2 2 0 012-2z" fill="none" stroke="currentColor" strokeWidth="1.6" />
      <path d="M14 2v5h5" fill="none" stroke="currentColor" strokeWidth="1.6" />
    </svg>
  );
}
