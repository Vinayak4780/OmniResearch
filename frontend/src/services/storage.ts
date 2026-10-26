/** localStorage access that never throws (private windows, blocked storage, SSR/tests). */

const ADMIN_TOKEN_KEY = 'omni.adminToken';
export const ADMIN_TOKEN_EVENT = 'omni:admin-token-changed';

export function safeGet(key: string): string | null {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

export function safeSet(key: string, value: string | null): boolean {
  try {
    if (value === null || value === '') window.localStorage.removeItem(key);
    else window.localStorage.setItem(key, value);
    return true;
  } catch {
    return false;
  }
}

export function getAdminToken(): string | null {
  const v = safeGet(ADMIN_TOKEN_KEY);
  return v && v.trim() ? v.trim() : null;
}

/** Returns false when the token could not be persisted (storage unavailable). */
export function setAdminToken(token: string | null): boolean {
  const ok = safeSet(ADMIN_TOKEN_KEY, token ? token.trim() : null);
  try {
    window.dispatchEvent(new Event(ADMIN_TOKEN_EVENT));
  } catch {
    /* ignore */
  }
  return ok;
}
